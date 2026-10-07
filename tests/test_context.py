import json
import re
import sys
import zipfile

import numpy as np

from core import fulltext, llm, writer
from core.db import connect, init_db, upsert_papers
from core.generate import generate
from core.related import related
from core.search_dense import DenseIndex
from tests.conftest import make_row


def _db(tmp_path, ids):
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(conn, [make_row(pid, f"Title {pid}", f"First sentence of {pid}. Second one. Third one.")
                         for pid in ids])
    return conn


def test_related_finds_nearest_and_excludes_seeds(tmp_path):
    conn = _db(tmp_path, ["2101.00001", "2101.00002", "2101.00003", "2101.00004"])
    pk = dict(conn.execute("SELECT id, pk FROM papers").fetchall())
    vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0.5, 0.5, 0], [0, 0, 1]], dtype=np.float16)
    pks = np.array([pk["2101.00001"], pk["2101.00002"], pk["2101.00003"], pk["2101.00004"]])
    index = DenseIndex(vectors, pks, {"model": "x"})
    found = related(conn, index, ["2101.00001"], k=2)
    assert [r["id"] for r in found] == ["2101.00002", "2101.00003"]
    assert related(conn, index, ["9999.99999"], k=2) == []


def test_generation_uses_context_only_in_allowed_sections(tmp_path, monkeypatch):
    prompts = []

    def complete(prompt, model, max_tokens):
        prompts.append(prompt)
        return ("T" if "Return only the title" in prompt else "Claim [1] and context [3]."), 10, "stop"

    monkeypatch.setattr(llm, "complete", complete)
    monkeypatch.setattr(llm, "chat", lambda prompt, model, **kw: (
        {"problem": "P.", "method": "M.", "setup": "S.", "findings": "F.", "limitations": "L."}, 5))
    conn = _db(tmp_path, ["2005.11401", "2005.11402", "2005.11403"])
    out = generate(conn, connect(tmp_path / "c.db"), ["2005.11401", "2005.11402"], doc_type="survey",
                   context_ids=["2005.11403", "2005.11401"],
                   fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    assert [c["id"] for c in out["context"]] == ["2005.11403"] and out["context"][0]["n"] == 3
    assert out["citations"]["invalid_numbers"] == []
    md = out["markdown"]
    assert "### Reviewed works" in md and "### Context works (summary only)" in md
    assert md.index("[2] ") < md.index("### Context works") < md.index("[3] ")
    intro = [p for p in prompts if 'Now write: the section "Introduction"' in p][0]
    per_source = [p for p in prompts if "the subsection about source" in p][0]
    assert "Context sources (known only from these short summaries)" in intro
    assert "First sentence of 2005.11403. Second one." in intro and "Third one" not in intro
    assert "Sources 3 to 3 are context sources" in intro
    assert "Context sources" not in per_source


def test_import_bundle_maps_ids_to_local_rows(tmp_path, monkeypatch):
    from scripts import import_bundle
    conn = _db(tmp_path, ["2101.00001", "2101.00002"])
    pk = dict(conn.execute("SELECT id, pk FROM papers").fetchall())
    conn.close()
    src = tmp_path / "bundle"
    (src / "models").mkdir(parents=True)
    np.save(src / "vectors.npy", np.eye(3, 4, dtype=np.float16))
    np.save(src / "ids.npy", np.array(["2101.00002", "2101.00001", "9999.99999"]))
    (src / "meta.json").write_text(json.dumps({"model": "granite-311m-384", "count": 3, "max_seq_length": 512}))
    for name in ("ranker.txt", "guard.joblib", "field_clf.joblib"):
        (src / "models" / name).write_text("x")
    bundle = tmp_path / "local_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(src))
    out = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["x", str(bundle), "--db", str(tmp_path / "papers.db"), "--out-dir", str(out)])
    import_bundle.main()
    index = DenseIndex.load(out / "vectors_dev")
    assert index.pks.tolist() == [pk["2101.00002"], pk["2101.00001"]] and index.meta["count"] == 2
    assert index.vectors[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert sorted(f.name for f in (out / "models").iterdir()) == ["field_clf.joblib", "guard.joblib", "ranker.txt"]
    assert not (out / "_bundle").exists()
    from core.search_bm25 import has_term_df
    assert has_term_df(connect(tmp_path / "papers.db"))
