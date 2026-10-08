import re

import numpy as np
import pytest

from core import citations, coverage, fulltext, llm, scholar, survey
from core.db import connect, init_db, upsert_papers
from core.search_dense import DenseIndex
from tests.conftest import make_row

A = [f"2101.0000{i}" for i in range(1, 7)]
B = [f"2102.0000{i}" for i in range(1, 7)]
TITLES = {A[0]: "GShard: Scaling Giant Models", A[1]: "GLaM: Efficient Scaling of Language Models",
          **{pid: f"Sparse expert routing study {i}" for i, pid in enumerate(A[2:], start=3)},
          **{pid: f"Dense retrieval for question answering {i}" for i, pid in enumerate(B, start=1)}}


REAL_BATCH, REAL_REFERENCES = scholar.batch, scholar.references


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(scholar, "references", lambda pid, limit=200: [])
    monkeypatch.setattr(scholar, "batch", lambda ids: {})


class FakePipeline:
    def __init__(self, index):
        self.index = index

    def search(self, text, k=10):
        if "pizza" in text:
            return {"results": [], "accepted": False}
        order = A + B if "expert" in text.lower() else B + A
        return {"results": [{"id": pid, "score": 1.0} for pid in order[:k]], "accepted": True}


def _world(tmp_path):
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(conn, [make_row(pid, TITLES[pid], f"Background sentence. We propose method {pid} with 5 experts. "
                                                    f"Closing remark.") for pid in A + B])
    pk = dict(conn.execute("SELECT id, pk FROM papers").fetchall())
    rng = np.random.default_rng(0)
    rows = [np.array([1, 0, 0, 0]) + rng.normal(0, 0.05, 4) for _ in A] + \
           [np.array([0, 1, 0, 0]) + rng.normal(0, 0.05, 4) for _ in B]
    vectors = np.array([r / np.linalg.norm(r) for r in rows], dtype=np.float16)
    index = DenseIndex(vectors, np.array([pk[p] for p in A + B]), {"model": "x"})
    return conn, FakePipeline(index)


def _fake_llm(monkeypatch, cite="GShard [1] and GLaM [2] work together [3]."):
    def chat(prompt, model, **kw):
        if "search queries" in prompt:
            return {"queries": ["sparse expert routing", "pizza place near me"]}, 30
        if "clusters of paper titles" in prompt:
            clusters = sorted(int(c) for c in re.findall(r"Cluster (\d+):", prompt))
            return {"themes": [{"cluster": c, "name": f"Theme {c}", "description": "d", "off_topic": False}
                               for c in clusters], "order": list(reversed(clusters))}, 40
        return {"problem": "P.", "method": "M.", "setup": "S.", "findings": "F.", "limitations": "L."}, 50

    def complete(prompt, model, max_tokens):
        return ("Review Title" if "Return only the title" in prompt else cite), 10, "stop"

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "complete", complete)


def test_key_sentences_prefer_contributions():
    text = "Scaling matters a lot. We propose GLaM with 1.2T parameters. It uses one third of the energy. Thanks."
    assert survey.key_sentences(text, n=2) == "We propose GLaM with 1.2T parameters."


def test_build_survey_end_to_end(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    conn, pipeline = _world(tmp_path)
    out = survey.build_broad(conn, connect(tmp_path / "c.db"), pipeline, "mixture of experts", seed_ids=[A[0]],
                              n_sources=10, max_deep=3, exclude=(B[5],),
                              fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    assert out["rejected_queries"] == ["pizza place near me"]
    assert B[5] not in out["candidate_ids"] and A[0] in out["screened_ids"]
    assert 1 <= len(out["themes"]) <= 6 and A[0] in out["deep"] and len(out["deep"]) <= 3
    assert all(t["size"] >= 3 for t in out["themes"]) and out["scholar"]["ok"]
    md = out["markdown"]
    refs = md.split("## References")[1].split("## Further reading")[0]
    numbers = [int(n) for n in re.findall(r"^\[(\d+)\]", refs, re.M)]
    assert numbers == list(range(1, len(numbers) + 1)) == list(range(1, len(out["cited"]) + 1))
    assert out["cited"][0] == A[0] and len(out["further_reading"]) == out["screened"] - len(out["cited"])
    assert "## Further reading" in md and "https://arxiv.org/abs/" in md.split("## Further reading")[1]
    assert md.startswith("# Review Title") and "## Comparison and Discussion" in md
    assert out["citations"]["invalid_numbers"] == [] and "name_mismatches" in out["citations"]


def test_off_topic_cluster_is_dropped_and_order_is_followed(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda prompt, model, **kw: (
        {"themes": [{"cluster": 0, "name": "Kept", "description": "", "off_topic": False},
                    {"cluster": 1, "name": "Noise", "description": "", "off_topic": True},
                    {"cluster": 2, "name": "Second", "description": ""}], "order": ["2", 0]}, 5))
    themes, _ = survey.name_themes("t", ["a", "b", "c"], np.array([0, 1, 2]), {"a": "A", "b": "B", "c": "C"}, "m")
    assert [t["name"] for t in themes] == ["Second", "Kept"]


def test_cluster_handles_tiny_input():
    assert survey.cluster(np.eye(3, dtype=np.float32)).tolist() == [0, 0, 0]


def test_renumber_by_first_appearance():
    text, mapping = citations.renumber("A [7]. B [3, 7]. C [5].")
    assert text == "A [1]. B [1, 2]. C [3]." and mapping == {7: 1, 3: 2, 5: 3}


def test_name_mismatch_and_meta_language():
    sources = [{"n": 1, "title": "GShard: Scaling Giant Models"}, {"n": 2, "title": "GLaM: Efficient MoE Scaling"},
               {"n": 3, "title": "ST-MoE: Stable MoE Models"}]
    text = "The original idea came from GShard [2]. GLaM scales [2]. Sparse MoE layers help [3]."
    report = citations.name_mismatches(text, sources)
    assert report["count"] == 1 and "GShard [2]" in report["examples"][0]
    assert citations.meta_language("The contextual source and its summary say so.")["count"] == 2
    assert citations.meta_language("Experts route tokens.")["count"] == 0


def test_coverage_from_reference_links():
    html = ('<a href="https://arxiv.org/abs/2101.03961v3">x</a> arXiv:1701.06538 '
            'https://arxiv.org/pdf/hep-th/9901001 and 2401.04088 without prefix')
    assert coverage.ids_in_text(html) == {"2101.03961", "1701.06538", "hep-th/9901001"}
    cov = coverage.coverage({"a", "b", "c", "d"}, ["a", "b", "c"], ["a", "b"], ["a"])
    assert (cov["candidates"], cov["screened"], cov["cited"]) == (0.75, 0.5, 0.25)


def test_no_candidates_gives_clear_error(tmp_path, monkeypatch):
    import pytest
    _fake_llm(monkeypatch)
    conn, pipeline = _world(tmp_path)
    pipeline.search = lambda text, k=10: {"results": [], "accepted": False}
    with pytest.raises(ValueError, match="yeterli makale"):
        survey.build_broad(conn, connect(tmp_path / "c.db"), pipeline, "pizza")


def _headings(md):
    return [line for line in md.splitlines() if line.startswith("## ")]


def test_proposal_uses_themes_as_related_work_and_writes_abstract_last(tmp_path, monkeypatch):
    targets = []
    _fake_llm(monkeypatch)
    original = llm.complete

    def complete(prompt, model, max_tokens):
        m = re.search(r"Now write: the section \"(.*)\"", prompt)
        if m:
            targets.append(m.group(1))
        return original(prompt, model, max_tokens)

    monkeypatch.setattr(llm, "complete", complete)
    conn, pipeline = _world(tmp_path)
    out = survey.build_broad(conn, connect(tmp_path / "c.db"), pipeline, "mixture of experts", "proposal",
                             seed_ids=[A[0]], lang="tr", n_sources=10, max_deep=2,
                             fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    heads = _headings(out["markdown"])
    assert heads[0] == "## Özet" and heads[1] == "## Problem ve Motivasyon"
    assert any(h.startswith("## İlgili Çalışmalar: Theme") for h in heads) and "## Kaynakça" in heads
    assert targets[-1] == "Özet" and out["doc_type"] == "proposal"
    assert "## İleri okuma" in heads


def test_synthesis_has_findings_by_theme_and_comparison_table_task(tmp_path, monkeypatch):
    prompts = []
    _fake_llm(monkeypatch)
    original = llm.complete
    monkeypatch.setattr(llm, "complete", lambda prompt, model, max_tokens: (
        prompts.append(prompt), original(prompt, model, max_tokens))[1])
    conn, pipeline = _world(tmp_path)
    out = survey.build_broad(conn, connect(tmp_path / "c.db"), pipeline, "mixture of experts", "synthesis",
                             n_sources=10, fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    heads = _headings(out["markdown"])
    assert heads[0] == "## Summary" and any(h.startswith("## Findings: Theme") for h in heads)
    agreement = [p for p in prompts if 'Now write: the section "Agreements and Differences"' in p][0]
    assert "markdown table" in agreement and "Never mention summaries" in agreement


def test_unknown_broad_type_is_rejected(tmp_path):
    import pytest
    conn = connect(tmp_path / "p.db")
    with pytest.raises(ValueError, match="Bilinmeyen tür"):
        survey.build_broad(conn, connect(tmp_path / "c.db"), None, "x", "poem")


def test_scholar_parses_batch_and_references(monkeypatch):
    def fake(url, payload=None, retries=5):
        if "batch" in url:
            assert payload == {"ids": ["ARXIV:2101.03961", "ARXIV:9999.99999"]}
            return [{"citationCount": 5000, "year": 2021}, None]
        return {"data": [{"citedPaper": {"externalIds": {"ArXiv": "1701.06538"}, "citationCount": 3000, "year": 2017}},
                         {"citedPaper": {"externalIds": {"DOI": "x"}, "citationCount": 9}}]}

    monkeypatch.setattr(scholar, "_request", fake)
    monkeypatch.setattr(scholar, "batch", REAL_BATCH)
    monkeypatch.setattr(scholar, "references", REAL_REFERENCES)
    assert scholar.batch(["2101.03961", "9999.99999"]) == {"2101.03961": {"citations": 5000, "year": 2021}}
    assert scholar.references("2101.03961") == [{"id": "1701.06538", "citations": 3000, "year": 2017}]


def test_citation_prior_reranks_and_references_expand(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    conn, pipeline = _world(tmp_path)
    monkeypatch.setattr(scholar, "references", lambda pid, limit=200: [
        {"id": B[4], "citations": 900, "year": 2020}, {"id": "0000.00000", "citations": 5, "year": 2020}])
    monkeypatch.setattr(scholar, "batch", lambda ids: {pid: {"citations": 4000 if pid == B[3] else 1, "year": 2024}
                                                        for pid in ids})
    pipeline.search = lambda text, k=10: {"results": [{"id": pid, "score": 1.0} for pid in A[:4]], "accepted": True}
    out = survey.build_broad(conn, connect(tmp_path / "c.db"), pipeline, "experts", seed_ids=[A[0]], n_sources=4,
                             fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    assert out["scholar"]["reference_additions"] == 1 and B[4] in out["candidate_ids"]
    assert "0000.00000" not in out["candidate_ids"]
    scores = survey.rerank({"x": 1.0, "y": 0.5}, survey.citation_scores(
        {"x": {"citations": 1, "year": 2025}, "y": {"citations": 5000, "year": 2025}}, 2026))
    assert scores["y"] > scores["x"]


def test_scholar_failure_does_not_stop_generation(tmp_path, monkeypatch):
    _fake_llm(monkeypatch)
    conn, pipeline = _world(tmp_path)

    def down(pid, limit=200):
        raise OSError("network unreachable")

    monkeypatch.setattr(scholar, "references", down)
    out = survey.build_broad(conn, connect(tmp_path / "c.db"), pipeline, "experts", n_sources=8,
                             fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    assert out["scholar"]["ok"] is False and "network unreachable" in out["scholar"]["error"]
    assert out["markdown"].startswith("# Review Title")


def test_survey_cap_moves_extra_surveys_to_further_reading():
    titles = {"a": "A Survey of X", "b": "Method B", "c": "X: A Review", "d": "An Overview of X", "e": "Method E"}
    scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
    screened, moved = survey.screen(scores, [], 4, titles, max_surveys=2)
    assert screened == ["a", "b", "c", "e"] and moved == ["d"]
    assert survey.is_survey("Retrieval-Augmented Generation: A Survey") and not survey.is_survey("Switch Transformers")


def test_small_clusters_merge_into_nearest():
    vectors = np.array([[1, 0], [0.9, 0.1], [0.95, 0.05], [0, 1], [0.1, 0.9], [0.05, 0.95], [0.8, 0.2]],
                       dtype=np.float32)
    labels = survey.merge_small(np.array([0, 0, 0, 1, 1, 1, 2]), vectors)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 0]


def test_number_audit_flags_numbers_missing_from_source():
    materials = {1: "GLaM has 1.2 trillion parameters, about 7x larger than GPT-3, and uses half the flops.",
                 2: "Training on 1,440 GPUs reached 1.41M tokens per second."}
    text = ("GLaM uses 1.2 trillion parameters [1]. It needs 14 percent of the compute [1]. "
            "MegaScale trains on 1,440 GPUs at 1.41 M tokens per second [2]. Both scale well [1, 2]. "
            "GPT-3 sized models matter [1].")
    report = citations.number_audit(text, materials)
    assert report["checked"] == 4 and report["unsupported"] == 1 and "[1] 14" in report["examples"][0]
