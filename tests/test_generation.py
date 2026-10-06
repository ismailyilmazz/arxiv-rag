import io
import json
import sys

import pytest

from core import cards, citations, fulltext, llm, writer
from core.db import connect, init_db, upsert_papers
from core.generate import generate
from tests.conftest import make_row

HTML = b"""<html><body><article class="ltx_document">
<div class="ltx_authors">A. Author</div>
<div class="ltx_abstract"><h6 class="ltx_title">Abstract</h6><p>We study sparse experts.</p></div>
<section class="ltx_section"><h2 class="ltx_title">1 Introduction</h2>
<div class="ltx_para"><p>Large models are costly <math alttext="O(n^2)">x</math> to run.</p></div></section>
<section class="ltx_section"><h2 class="ltx_title">2 Method</h2>
<div class="ltx_para"><p>We route tokens to experts.</p></div>
<figure><figcaption>Figure 1</figcaption></figure></section>
<section class="ltx_section"><h2 class="ltx_title">3 Conclusion</h2><p>Experts help.</p></section>
<section class="ltx_bibliography"><h2>References</h2><p>[1] Someone.</p></section>
</article></body></html>"""

ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2401.04088v1</id><published>2024-01-08T18:47:34Z</published>
<title>Mixtral of
 Experts</title><summary>We introduce Mixtral.</summary>
<author><name>Albert Q. Jiang</name></author><author><name>Alexandre Sablayrolles</name></author>
<arxiv:primary_category term="cs.LG"/></entry>
<entry><id>http://arxiv.org/abs/math/0211159v1</id><published>2002-11-11T16:11:49Z</published>
<title>The entropy formula for the Ricci flow</title><summary>We present a monotonic expression.</summary>
<author><name>Grisha Perelman</name></author><arxiv:primary_category term="math.DG"/></entry>
</feed>"""


def make_pdf(lines):
    content = "BT /F1 12 Tf 72 740 Td 14 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


PDF = make_pdf(["Retrieval Paper", "Abstract", "We retrieve.", "1 Introduction", "Knowledge is stored in weights.",
                "2 Method", "We combine a retriever and a generator.", "3 Conclusion", "Retrieval helps.",
                "References", "[1] Old work."])


def test_parse_html_keeps_sections_and_math_drops_bibliography():
    sections = fulltext.parse_html(HTML)
    titles = [t for t, _ in sections]
    assert titles == ["Abstract", "1 Introduction", "2 Method", "3 Conclusion"]
    text = dict(sections)
    assert "O(n^2)" in text["1 Introduction"]
    assert "Figure 1" not in text["2 Method"] and "Someone" not in " ".join(text.values())


def test_parse_pdf_splits_sections_and_stops_at_references():
    sections = dict(fulltext.parse_pdf(PDF))
    assert sections["2 Method"] == "We combine a retriever and a generator."
    assert "Old work" not in " ".join(sections.values())


@pytest.mark.parametrize("responses,expected", [
    ({"html": (200, HTML)}, "html"),
    ({"html": (404, b""), "pdf": (200, PDF)}, "pdf"),
    ({"html": (404, b""), "pdf": (404, b"")}, "abstract"),
])
def test_fetch_prefers_html_then_pdf_then_abstract(monkeypatch, responses, expected):
    def fake_get(url):
        kind = "html" if "/html/" in url else "pdf"
        return responses.get(kind, (404, b""))
    monkeypatch.setattr(fulltext, "_get", fake_get)
    assert fulltext.fetch("2401.04088").source == expected


def test_requests_wait_between_calls(monkeypatch):
    clock, sleeps = [100.0], []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    monkeypatch.setattr(fulltext.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fulltext.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(fulltext.urllib.request, "urlopen", lambda req, timeout: Response())
    monkeypatch.setattr(fulltext, "_last_request", [0.0])
    fulltext._get("https://arxiv.org/html/x")
    clock[0] += 1.0
    fulltext._get("https://arxiv.org/html/y")
    assert sleeps == [pytest.approx(2.0)]


def test_select_text_uses_abstract_and_key_sections_within_budget():
    sections = [("Abstract", "dup"), ("1 Introduction", "i" * 5000), ("2 Related Work", "r"),
                ("3 Proposed Approach", "m"), ("4 Conclusion", "c")]
    text = fulltext.select_text(sections, "The abstract.")
    assert text.startswith("Abstract: The abstract.")
    assert "1 Introduction: " + "i" * fulltext.BUDGET["introduction"] + "\n" in text
    assert "3 Proposed Approach: m" in text and "4 Conclusion: c" in text and "Related Work" not in text


def test_fetch_metadata_parses_atom(monkeypatch):
    monkeypatch.setattr(fulltext, "_get", lambda url: (200, ATOM))
    meta = fulltext.fetch_metadata(["2401.04088", "math/0211159"])
    assert meta["2401.04088"]["title"] == "Mixtral of Experts"
    assert meta["2401.04088"]["authors"] == "Albert Q. Jiang, Alexandre Sablayrolles"
    assert meta["math/0211159"]["published"] == "2002-11-11"


def test_citation_numbers_and_fix():
    assert citations.numbers_in("1, 3-4") == [1, 3, 4]
    text = "A [1]. B [2, 7]. C [9]. D [1\u20132].\n\n## References\n[1] made up"
    fixed, report = citations.check_and_fix(text, 3)
    assert fixed == "A [1]. B [2]. C . D [1, 2]."
    assert report["invalid_numbers"] == [7, 9]
    assert report["uncited_sources"] == [3]
    assert report["removed_model_references"] is True


def test_bibliography_is_built_from_metadata():
    sources = [{"n": 1, "id": "2401.04088", "title": "Mixtral of Experts", "published": "2024-01-08",
                "authors": "A. Jiang, B. Sablayrolles, C. Mensch and D. Bamford"}]
    bib = citations.bibliography(sources, "tr")
    assert bib.startswith("## Kaynakça")
    assert "[1] A. Jiang, B. Sablayrolles, C. Mensch et al. (2024). Mixtral of Experts. arXiv:2401.04088. " \
           "https://arxiv.org/abs/2401.04088" in bib


def test_prompt_follows_type_and_language():
    sources = [{"n": 1, "title": "T", "published": "2020", "card": dict.fromkeys(cards.FIELDS, "x")}]
    prompt = writer.build_prompt(sources, "proposal", "tr")
    assert "research proposal in Turkish" in prompt and "## Önerilen Yöntem" in prompt
    assert "Use only the numbers 1 to 1" in prompt


def _fake_llm(monkeypatch, calls):
    def chat(prompt, model, **kw):
        calls.append(("card", model))
        return {"problem": "P.", "method": "M.", "setup": "S.", "findings": "F.", "limitations": "L."}, 100

    def complete(prompt, model, **kw):
        calls.append(("write", model))
        return "# Title\n\n## Introduction\nClaim [1] and [2]. Wrong [5].", 500, "stop"

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "complete", complete)


def test_generate_end_to_end_and_cache(tmp_path, monkeypatch):
    calls, fetched = [], []
    _fake_llm(monkeypatch, calls)
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(conn, [make_row("2005.11401", "Retrieval-Augmented Generation", "We combine retrieval.")])
    cache = connect(tmp_path / "cache.db")

    def fetcher(pid):
        fetched.append(pid)
        return fulltext.FullText(pid, "html", [("1 Introduction", "intro")])

    meta = lambda ids: {"2401.04088": {"title": "Mixtral of Experts", "abstract": "We introduce Mixtral.",
                                        "authors": "Albert Q. Jiang", "published": "2024-01-08",
                                        "primary_category": "cs.LG", "license": None}}
    out = generate(conn, cache, ["2005.11401", "2401.04088v1"], doc_type="survey", lang="en",
                   card_model="small", write_model="big", fetcher=fetcher, metadata_fetcher=meta)
    assert out["citations"]["invalid_numbers"] == [5]
    assert "[5]" not in out["markdown"]
    assert out["markdown"].rstrip().endswith("[2] Albert Q. Jiang (2024). Mixtral of Experts. arXiv:2401.04088. "
                                             "https://arxiv.org/abs/2401.04088")
    assert out["tokens"] == {"cards": 200, "write": 500, "total": 700}
    assert [m for k, m in calls if k == "card"] == ["small", "small"] and ("write", "big") in calls

    again = generate(conn, cache, ["2005.11401", "2401.04088"], doc_type="synthesis", lang="tr",
                     card_model="small", write_model="big", fetcher=fetcher, metadata_fetcher=meta)
    assert again["tokens"]["cards"] == 0 and all(s["cached"] for s in again["sources"])
    assert fetched == ["2005.11401", "2401.04088"]
    assert "## Kaynakça" in again["markdown"]


def test_cc_policy_uses_abstract_for_closed_license(tmp_path, monkeypatch):
    calls = []
    _fake_llm(monkeypatch, calls)
    cache = connect(tmp_path / "cache.db")
    cards.init_cache(cache)
    paper = {"id": "2005.11401", "title": "T", "abstract": "A", "license": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"}
    card, meta = cards.get_card(cache, paper, "m", policy="cc", fetcher=lambda pid: pytest.fail("fetch edilmemeli"))
    assert meta["source"] == "abstract" and card["problem"] == "P."


def test_generate_rejects_bad_input(tmp_path):
    conn, cache = connect(tmp_path / "p.db"), connect(tmp_path / "c.db")
    init_db(conn)
    with pytest.raises(ValueError):
        generate(conn, cache, ["2005.11401"], doc_type="poem")
    with pytest.raises(ValueError):
        generate(conn, cache, [f"2005.1140{i}" for i in range(6)])


def test_cli_writes_markdown_and_report(tmp_path, monkeypatch):
    from scripts import generate_paper
    calls = []
    _fake_llm(monkeypatch, calls)
    db = tmp_path / "papers.db"
    conn = connect(db)
    init_db(conn)
    upsert_papers(conn, [make_row("2005.11401", "RAG", "We combine retrieval.")])
    conn.close()
    monkeypatch.setattr(fulltext, "fetch", lambda pid: fulltext.FullText(pid, "pdf", [("2 Method", "m")]))
    monkeypatch.setattr(fulltext, "_get", lambda url: pytest.fail("ağa çıkılmamalı"))
    out_dir = tmp_path / "gen"
    monkeypatch.setattr(sys, "argv", ["x", "--ids", "2005.11401", "--type", "synthesis", "--db", str(db),
                                      "--cache-db", str(tmp_path / "c.db"), "--out-dir", str(out_dir)])
    generate_paper.main()
    report = json.loads(next(out_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["doc_type"] == "synthesis" and report["lang"] == "en"
    assert report["sources"][0]["text_source"] == "pdf"
    assert next(out_dir.glob("*.md")).read_text(encoding="utf-8").startswith("# Title")


def test_uncited_paragraphs_are_counted():
    long_cited = " ".join(["word"] * 30) + " [1]."
    long_uncited = " ".join(["word"] * 30) + "."
    text = f"# T\n\n## A\n\n{long_cited}\n\n{long_uncited}\n\n- {long_uncited}\n- short bullet [1]\n\nShort line."
    _, report = citations.check_and_fix(text, 1)
    assert report["paragraphs"] == 3 and report["uncited_paragraphs"] == 2


def test_prompt_asks_for_needed_length_without_fixed_word_count():
    sources = [{"n": 1, "title": "T", "published": "2020", "card": dict.fromkeys(cards.FIELDS, "x")}]
    prompt = writer.build_prompt(sources, "synthesis", "en")
    assert "Every paragraph and every bullet point" in prompt
    assert "as long as the sources support, and no longer" in prompt and "Never pad" in prompt
    assert " words" not in prompt.split("Rules:")[1].replace("words, ", "")


def test_output_budget_uses_all_room_under_minute_limit():
    assert writer.output_budget("x" * 3000) == writer.TOKEN_BUDGET - 1000
    assert writer.output_budget("x" * 30000) == 1500


def test_old_cards_are_regenerated_after_version_change(tmp_path, monkeypatch):
    calls = []
    _fake_llm(monkeypatch, calls)
    cache = connect(tmp_path / "cache.db")
    cache.execute("CREATE TABLE cards (paper_id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT NOT NULL, "
                  "card TEXT NOT NULL, tokens INTEGER NOT NULL, created_at TEXT NOT NULL)")
    cache.execute("INSERT INTO cards VALUES ('2005.11401', 'html', 'm', '{\"problem\": \"old\"}', 10, 't')")
    cards.init_cache(cache)
    paper = {"id": "2005.11401", "title": "T", "abstract": "A", "license": None}
    card, meta = cards.get_card(cache, paper, "m", fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    assert meta["cached"] is False and card["setup"] == "S."
    card, meta = cards.get_card(cache, paper, "m", fetcher=lambda pid: pytest.fail("önbellekten gelmeli"))
    assert meta["cached"] is True


def test_complete_shrinks_request_when_too_large(monkeypatch):
    requested = []

    class Choice:
        finish_reason = "length"
        message = type("M", (), {"content": " text "})()

    class Completions:
        def create(self, **kw):
            requested.append(kw["max_tokens"])
            if len(requested) == 1:
                raise Exception("Request too large for model on tokens per minute (TPM): Limit 8000, Requested 9500")
            return type("R", (), {"choices": [Choice()], "usage": type("U", (), {"total_tokens": 42})()})()

    client = type("C", (), {"chat": type("Ch", (), {"completions": Completions()})()})()
    monkeypatch.setattr(llm, "_client", lambda: client)
    text, tokens, finish = llm.complete("p", "openai/gpt-oss-120b", max_tokens=5000)
    assert requested == [5000, 3300] and (text, tokens, finish) == ("text", 42, "length")


def test_truncation_is_reported(tmp_path, monkeypatch):
    calls = []
    _fake_llm(monkeypatch, calls)
    monkeypatch.setattr(llm, "complete", lambda prompt, model, **kw: ("# T\n\nCut off [1", 900, "length"))
    conn, cache = connect(tmp_path / "p.db"), connect(tmp_path / "c.db")
    init_db(conn)
    upsert_papers(conn, [make_row("2005.11401", "RAG", "We combine retrieval.")])
    out = generate(conn, cache, ["2005.11401"], fetcher=lambda pid: fulltext.FullText(pid, "html", []))
    assert out["length"]["truncated"] is True and out["length"]["output_budget_tokens"] >= 1500
