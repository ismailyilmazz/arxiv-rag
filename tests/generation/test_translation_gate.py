import pytest

from arxiv_rag.generation import language, llm, writer

EN = (
    "Hallucinations in large language models (LLMs) jeopardize the trustworthiness of automated text generation "
    "and raise ethical concerns when fabricated statements are presented as factual knowledge [1]."
)
TR = (
    "Büyük dil modellerindeki halüsinasyonlar, otomatik metin üretiminin güvenilirliğini tehlikeye atar ve uydurma "
    "ifadeler gerçek bilgi gibi sunulduğunda etik kaygılara yol açar [1]."
)
TABLE = "| Work | Year | Key idea |\n|---|---|---|\n| HaluEval | 2023 | sampling then filtering for the benchmark |"


def test_english_share_counts_prose_paragraphs_only():
    assert language.english_share(EN) == 1.0 and language.english_share(TR) == 0.0
    assert language.english_share(f"{TR}\n\n{EN}") == 0.5
    assert language.english_share(f"{TR}\n\n{TABLE}") == 0.0
    assert language.audit(f"{TR}\n\n{EN}\n\n{EN}") == {"paragraphs": 3, "english_paragraphs": 2, "english_share": 0.667}


class Script:
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt, model, max_tokens, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0), 100, "stop"


def test_good_translation_needs_one_call(monkeypatch):
    fake = Script(TR)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, truncated, attempts = writer.translate_section(EN, "survey", "tr", "m")
    assert text == TR and attempts == 1 and tokens == 100 and not truncated


def test_english_reply_is_retried_with_a_stricter_prompt(monkeypatch):
    fake = Script(EN, TR)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, _, attempts = writer.translate_section(EN, "survey", "tr", "m")
    assert text == TR and attempts == 2 and tokens == 200
    assert fake.prompts[1].startswith("The previous translation attempt returned English")


def test_only_remaining_english_paragraphs_are_repaired(monkeypatch):
    fake = Script(f"{EN}\n\n{EN}", TR, TR)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, _, attempts = writer.translate_section(f"{EN}\n\n{EN}", "survey", "tr", "m")
    assert attempts == 2 and tokens == 300 and language.english_share(text) == 0.0 and text.count(TR) == 2


EN4 = EN.replace("[1]", "[4]")
EN5 = EN.replace("[1]", "[5]")
TR4 = TR.replace("[1]", "[4]")
TR5 = TR.replace("[1]", "[5]")


def test_lossy_translation_falls_back_to_paragraphs(monkeypatch):
    fake = Script(TR, TR, TR4, TR5)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, _, attempts = writer.translate_section(f"{EN}\n\n{EN4}\n\n{EN5}", "survey", "tr", "m")
    assert attempts == 2 and tokens == 400 and text == f"{TR}\n\n{TR4}\n\n{TR5}"


def test_mixed_section_keeps_turkish_paragraphs_untouched(monkeypatch):
    fake = Script(TR)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, repaired, failed = writer.repair_paragraphs(f"{EN}\n\n{TR4}\n\n{TR5}", "tr", "m")
    assert len(fake.prompts) == 1 and EN in fake.prompts[0]
    assert text == f"{TR}\n\n{TR4}\n\n{TR5}" and (repaired, failed) == (1, 0)


LOST = "Bu Türkçe metin atıfı kaybetmiş ama uzunluğu yeterli görünüyor ve çeviri gibi duruyor."


def test_repair_that_drops_a_citation_twice_keeps_the_original(monkeypatch):
    fake = Script(LOST, LOST)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, repaired, failed = writer.repair_paragraphs(EN, "tr", "m")
    assert text == EN and (repaired, failed) == (0, 1) and tokens == 200


def test_failed_repair_gets_one_more_chance(monkeypatch):
    fake = Script(LOST, TR)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, repaired, failed = writer.repair_paragraphs(EN, "tr", "m")
    assert text == TR and (repaired, failed) == (1, 0) and tokens == 200


def test_completeness_needs_all_citations_and_enough_length():
    assert writer.is_complete(TR, EN) and not writer.is_complete(TR, f"{EN} [2]")
    assert not writer.is_complete("Kısa [1].", EN)


def test_english_target_is_not_checked(monkeypatch):
    fake = Script(EN)
    monkeypatch.setattr(llm, "complete", fake)
    assert writer.translate_section(EN, "survey", "en", "m")[3] == 1


@pytest.mark.parametrize(
    "note",
    [
        " (Metnin geri kalan kısmı eksik olduğu için çevirisi tamamlanamamıştır.)",
        " (Sonuç kısmı eksik olduğu için çevirisi yapılamamıştır)",
        " (The rest could not be translated because the text is incomplete.)",
    ],
)
def test_translator_notes_are_removed(note):
    assert writer.strip_meta_notes(TR + note) == TR


def test_ordinary_parentheses_survive():
    text = "Model (eksik veri ile eğitilmiş) iyi çalışır; çeviri (makine çevirisi) görevi zordur."
    assert writer.strip_meta_notes(text) == text


def test_theme_names_are_translated(monkeypatch):
    monkeypatch.setattr(llm, "complete", Script("Çeviriler:\n1. **Yöntemler**\n2. Benchmarklar"))
    assert writer.translate_lines(["Methods", "Benchmarks"], "tr", "m") == (["Yöntemler", "Benchmarklar"], 100)
    assert writer.translate_lines(["Methods"], "en", "m") == (["Methods"], 0)


def test_heading_count_mismatch_falls_back_to_one_by_one(monkeypatch):
    fake = Script("Yöntemler", "Yöntemler", "")
    monkeypatch.setattr(llm, "complete", fake)
    names, tokens = writer.translate_lines(["Methods", "Benchmarks"], "tr", "m")
    assert names == ["Yöntemler", "Benchmarks"] and tokens == 300 and len(fake.prompts) == 3
