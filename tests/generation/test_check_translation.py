from arxiv_rag.generation import writer
from scripts.generation import check_translation

EN = "Hallucinations in large language models jeopardize the trustworthiness of the text and raise concerns [1]."
TR = "Büyük dil modellerindeki halüsinasyonlar metnin güvenilirliğini tehlikeye atar ve kaygılara yol açar [1]."
MD = (
    f"# Başlık\n\n## Giriş\n\n{TR}\n\n## Arka Plan\n\n{EN}\n\n## Kaynakça\n\n[1] A paper on the topic of the survey (2024).\n"
    "\n## İleri okuma\n\nA survey of hallucination in large language models and their many detectors (2023).\n"
)


def test_measure_only_changes_nothing():
    fixed, rows = check_translation.check(MD)
    assert fixed == MD and [(r["section"], r["before"]) for r in rows] == [("Giriş", 0.0), ("Arka Plan", 1.0)]
    assert "İleri okuma" not in [r["section"] for r in rows]


def test_fix_retranslates_only_english_sections(monkeypatch):
    calls = []
    monkeypatch.setattr(
        writer, "repair_paragraphs", lambda text, lang, model, **kw: (calls.append((text, kw)), (TR, 50, 1, 0))[1]
    )
    fixed, rows = check_translation.check(MD, fix=True, model="m")
    assert calls == [(EN, {"titles": []})]
    assert fixed.count(TR) == 2 and EN not in fixed
    assert "## Kaynakça\n\n[1] A paper on the topic" in fixed and rows[1]["after"] == 0.0 and rows[1]["repaired"] == 1


def test_fix_translates_english_theme_headings(monkeypatch):
    md = f"# Başlık\n\n## Broad Surveys of Methods\n\n{TR}\n\n## Kaynakça\n\n[1] A paper (2024).\n"
    monkeypatch.setattr(writer, "translate_lines", lambda lines, lang, model: (["Yöntemlere Genel Bakışlar"], 30))
    fixed, rows = check_translation.check(md, fix=True, model="m")
    assert "## Yöntemlere Genel Bakışlar\n" in fixed and "Broad Surveys" not in fixed
    assert rows[0]["heading_english"] is False and rows[0]["tokens"] == 30
    measured, rows = check_translation.check(md)
    assert measured == md and rows[0]["heading_english"] is True


def test_titles_are_read_from_references_and_further_reading():
    md = (
        "## Kaynakça\n\n[1] Ann Lee, Bo Kim (2003). Quantum cryptography with and without entanglement. "
        "arXiv:quant-ph/0312011. https://arxiv.org/abs/quant-ph/0312011\n\n## İleri okuma\n\n"
        "[2] Paneru (2019). Entanglement: Quantum or Classical? arXiv:1911.02201. https://arxiv.org/abs/1911.02201\n"
        "[3] Paneru (2019). Old Format Title?. arXiv:1911.02202. https://arxiv.org/abs/1911.02202\n\n"
        "- Quantum Cryptography: A Survey (n.d.). https://arxiv.org/abs/2003.07907\n"
    )
    assert check_translation.titles_in(md) == [
        "Quantum cryptography with and without entanglement",
        "Entanglement: Quantum or Classical?",
        "Old Format Title?",
        "Quantum Cryptography: A Survey",
    ]
