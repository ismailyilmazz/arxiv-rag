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
    monkeypatch.setattr(writer, "repair_paragraphs", lambda text, lang, model: (calls.append(text), (TR, 50, 1, 0))[1])
    fixed, rows = check_translation.check(MD, fix=True, model="m")
    assert calls == [EN] and fixed.count(TR) == 2 and EN not in fixed
    assert "## Kaynakça\n\n[1] A paper on the topic" in fixed and rows[1]["after"] == 0.0 and rows[1]["repaired"] == 1
