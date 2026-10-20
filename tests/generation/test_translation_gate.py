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
    found = language.audit(f"{TR}\n\n{EN}\n\n{EN}")
    assert (found["paragraphs"], found["english_paragraphs"], found["english_share"]) == (3, 2, 0.667)


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


EN_TABLE = (
    "| Work | Year | Key idea | Main result |\n|---|---|---|---|\n"
    "| HaluEval [1] | 2023 | Sampling then filtering of the answers | Detects most of the hallucinations |"
)
TR_TABLE = (
    "| Çalışma | Yıl | Ana fikir | Temel sonuç |\n|---|---|---|---|\n"
    "| HaluEval [1] | 2023 | Cevapları örnekleyip süzme | Halüsinasyonların çoğunu yakalar |"
)


def test_tables_headings_and_short_lines_are_measured():
    text = (
        f"## Arka Plan\n\n{TR}\n\n**Comparison of Representative Works**\n\n{EN_TABLE}\n\n## Broad Surveys of Methods"
    )
    found = language.audit(text)
    assert (found["paragraphs"], found["english_paragraphs"]) == (1, 0)
    assert (found["tables"], found["english_tables"]) == (1, 1)
    assert (found["lines"], found["english_lines"]) == (3, 2)
    assert language.audit(f"{TR}\n\n**Temsilî Çalışmalar**\n\n{TR_TABLE}")["english_tables"] == 0


@pytest.mark.parametrize("line", ["Arka Plan", "Kuantum Kriptografi Protokolleri", "BB84 ve QKD", "İkinci tema"])
def test_turkish_lines_without_turkish_letters_are_not_english(line):
    assert not language.text_is_english(line)


@pytest.mark.parametrize("line", ["Broad Surveys of Quantum Cryptography", "Entanglement‑Based QKD Security"])
def test_english_headings_are_english(line):
    assert language.text_is_english(line)


def test_english_table_left_by_the_translation_is_repaired(monkeypatch):
    fake = Script(f"{TR}\n\n{EN_TABLE}", TR_TABLE)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, _, attempts = writer.translate_section(f"{EN}\n\n{EN_TABLE}", "survey", "tr", "m")
    assert text == f"{TR}\n\n{TR_TABLE}" and attempts == 2 and tokens == 200
    assert (
        EN_TABLE.splitlines()[2] in fake.prompts[1] and "| Çalışma | Yıl | Ana fikir | Ana sonuç |" in fake.prompts[1]
    )
    assert "table cell" in fake.prompts[1]


def test_translation_prompt_asks_for_table_cells():
    prompt = writer.TRANSLATE_PROMPT.format(label="x", language="Turkish", text="t", glossary="")
    assert "table cells" in prompt and "unchanged" not in prompt


class Chat:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def __call__(self, prompt, model, **kw):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, 70


SECTIONS = [
    {"key": "intro", "heading": "Introduction", "heading_out": "Giriş", "text": EN},
    {"key": "theme:Broad Surveys", "heading": "Broad Surveys", "heading_out": "Genel Derlemeler", "text": EN},
]


def test_glossary_is_built_once_and_used_in_every_translation(monkeypatch):
    chat = Chat({"terms": [{"en": "hallucinations", "target": "halüsinasyonlar"}, {"en": "LLM", "target": "BDM"}]})
    fake = Script(TR, TR)
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm, "complete", fake)
    texts, rows, tokens, glossary = writer.translate_sections(
        SECTIONS, "survey", "tr", "m", {"Broad Surveys": "Genel Derlemeler"}
    )
    assert glossary == {"hallucinations": "halüsinasyonlar", "Broad Surveys": "Genel Derlemeler"}
    assert len(chat.prompts) == 1 and "hallucinations" in chat.prompts[0]
    assert all(
        "hallucinations = halüsinasyonlar" in p and "Broad Surveys = Genel Derlemeler" in p for p in fake.prompts
    )
    assert rows[0]["section"] == "translate:glossary" and rows[0]["terms"] == 2 and tokens == 70 + 200
    assert texts == {"intro": TR, "theme:Broad Surveys": TR} and rows[1]["english_left"] == 0


def test_failed_glossary_call_keeps_theme_names(monkeypatch):
    monkeypatch.setattr(llm, "chat", Chat(ValueError("bozuk JSON")))
    glossary, tokens = writer.build_glossary([EN, EN], {"Methods": "Yöntemler", "BB84": "BB84"}, "x", "tr", "m")
    assert glossary == {"Methods": "Yöntemler"} and tokens == 0


def test_heading_returned_unchanged_is_asked_again(monkeypatch):
    fake = Script("Broad Surveys of Methods", "Yöntemlere Genel Bakışlar")
    monkeypatch.setattr(llm, "complete", fake)
    names, tokens = writer.translate_lines(["Broad Surveys of Methods"], "tr", "m")
    assert names == ["Yöntemlere Genel Bakışlar"] and tokens == 200
    assert fake.prompts[1].startswith("Translate this heading")


def test_implausible_heading_reply_is_rejected(monkeypatch):
    monkeypatch.setattr(
        llm, "complete", Script("Methods", "Bu çalışmalar [1] ve [2] yöntemleri ayrıntılı biçimde inceler.")
    )
    assert writer.translate_lines(["Methods"], "tr", "m")[0] == ["Methods"]


def test_plan_names_are_used_and_only_missing_ones_are_translated(monkeypatch):
    fake = Script("Yöntemler")
    monkeypatch.setattr(llm, "complete", fake)
    themes = [{"name": "Surveys", "name_local": "Derlemeler"}, {"name": "Methods", "name_local": "Methods"}]
    local, tokens, from_plan = writer.local_names(themes, "tr", "m")
    assert local == {"Surveys": "Derlemeler", "Methods": "Yöntemler"} and from_plan == 1 and len(fake.prompts) == 1


def test_term_audit_ignores_titles_and_the_work_column():
    text = (
        "**Quantum Entanglement Survey** çalışması entanglement kavramını inceler [1].\n\n"
        "| Work | Ana fikir |\n|---|---|\n| Entanglement survey [2] | entanglement dağıtımı |"
    )
    found = language.term_audit(text, {"entanglement": "dolanıklık", "QKD": "QKD", "survey": "derleme"})
    assert found == {"terms": 3, "english_left": 2, "examples": {"entanglement": 2}}


def test_frequent_terms_prefers_repeated_technical_phrases():
    text = "Key distribution is secure. The key distribution relies on entanglement. Entanglement helps [1]."
    terms = language.frequent_terms(text)
    assert terms[0] == "key distribution" and "entanglement" in terms and "the" not in terms


TITLE = "Quantum Cryptography with Entangled Photons"
EN_T = f"The work **{TITLE}** [1] demonstrates a practical system over a free-space link of the city."
TR_T = f"**{TITLE}** [1] çalışması, şehirdeki serbest alan bağlantısı üzerinde pratik bir sistem gösterir."
TR_BAD = (
    "**Dolanık Fotonlarla Kuantum Kriptografi** [1] çalışması, şehirdeki serbest alan bağlantısında sistem gösterir."
)


def test_bold_and_italic_titles_are_left_out_of_the_measurement():
    found = language.audit(f"{TR_T}\n\n**{TITLE}**\n\n*Cryptography in a Quantum World* [3]")
    assert found["english_paragraphs"] == 0 and found["english_lines"] == 0 and found["lines"] == 0


def test_translated_title_is_rejected_and_the_paragraph_is_repaired(monkeypatch):
    fake = Script(TR_BAD, TR_T)
    monkeypatch.setattr(llm, "complete", fake)
    text, tokens, _, attempts = writer.translate_section(EN_T, "survey", "tr", "m", titles=[TITLE])
    assert text == TR_T and attempts == 2 and len(fake.prompts) == 2


def test_repair_that_translates_a_title_keeps_the_original(monkeypatch):
    monkeypatch.setattr(llm, "complete", Script(TR_BAD, TR_BAD))
    text, _, repaired, failed = writer.repair_paragraphs(EN_T, "tr", "m", titles=[TITLE])
    assert text == EN_T and (repaired, failed) == (0, 1)


def test_generic_title_words_outside_bold_are_free_to_translate(monkeypatch):
    en = "Quantum cryptography protocols rely on the laws of physics and on the assumptions of the model [2]."
    tr = "Kuantum kriptografi protokolleri fizik yasalarına ve modelin varsayımlarına dayanır [2]."
    monkeypatch.setattr(llm, "complete", Script(tr))
    assert writer.translate_section(en, "survey", "tr", "m", titles=["Quantum Cryptography"])[3] == 1


def test_theme_label_that_contains_a_short_title_is_not_protected():
    source = "*The second theme, Broad Surveys of Quantum Cryptography*, covers reviews [4]."
    assert writer.protected_titles(source, ["Quantum Cryptography"]) == set()
    assert writer.protected_titles(f"See **{TITLE}** [1].", [TITLE]) == {TITLE.lower()}


def test_work_column_keeps_the_original_titles():
    source = f"| Work | Year | Key idea |\n|---|---|---|\n| {TITLE} [1] | 1999 | Entangled photons over free space |"
    translated = (
        "| Çalışma | Yıl | Ana fikir |\n|---|---|---|\n"
        "| Dolanık Fotonlarla Kuantum Kriptografi [1] | 1999 | Serbest alanda dolanık fotonlar |"
    )
    kept = writer.keep_work_column(translated, source)
    assert kept.splitlines()[0] == "| Çalışma | Yıl | Ana fikir |"
    assert kept.splitlines()[2] == f"| {TITLE} [1] | 1999 | Serbest alanda dolanık fotonlar |"
    other = "| Dataset | Size |\n|---|---|\n| Long English cell | 10 |"
    assert writer.keep_work_column("| Veri | Boyut |\n|---|---|\n| Uzun hücre | 10 |", other).endswith(
        "Uzun hücre | 10 |"
    )


def test_translation_prompts_protect_titles():
    prompt = writer.TRANSLATE_PROMPT.format(label="x", language="Turkish", text="t", glossary="")
    retry = writer.RETRY_TRANSLATE_PROMPT.format(language="Turkish", text="t", glossary="")
    assert "Paper titles are never translated" in prompt and "Work column" in prompt and "Work column" in retry
    assert writer.glossary_rule({"survey": "derleme"}).startswith("\n- Outside paper titles")


@pytest.mark.parametrize(
    "name, words",
    [
        ("Entanglement'in Kuantum Kriptografisinde Kullanımı", ["Entanglement"]),
        ("Post‑Quantum ve Kuantum‑Dirençli Kriptografi", ["Quantum"]),
        ("Bell'in Eşitsizliği ve QKD Deneyleri", []),
        ("Kuantum Güvenli Primitiflerin Temelleri", []),
    ],
)
def test_english_words_in_turkish_names(name, words):
    assert language.english_words(name) == words


def test_names_with_english_words_are_translated_again(monkeypatch):
    fake = Script("Entanglement'in Kuantum Kriptografisinde Kullanımı", "Kuantum Kriptografisinde Dolanıklık")
    monkeypatch.setattr(llm, "complete", fake)
    themes = [{"name": "Entanglement in Quantum Cryptography", "name_local": "Entanglement'in Kullanımı"}]
    local, tokens, from_plan = writer.local_names(themes, "tr", "m")
    assert local == {"Entanglement in Quantum Cryptography": "Kuantum Kriptografisinde Dolanıklık"}
    assert from_plan == 0 and tokens == 200 and fake.prompts[1].startswith("Translate this heading")


def test_known_table_headers_are_localized_and_unknown_ones_are_measured():
    table = "| Work | Year | Key idea | Setting or scale | Main result |\n|---|---|---|---|---|\n| T [1] | 1999 | a | b | c |"
    assert writer.localize_table_headers(table, "tr").splitlines()[0] == (
        "| Çalışma | Yıl | Ana fikir | Ortam ya da ölçek | Ana sonuç |"
    )
    assert writer.localize_table_headers(table, "en") == table
    odd = "| Dataset | Evaluation setting | Limitations |\n|---|---|---|\n| T [1] | Laboratuvar ölçeği | Küçük veri kümesi |"
    found = language.audit(odd)
    assert (found["english_tables"], found["english_lines"]) == (0, 1)


def test_glossary_adds_fixed_terms_and_caps_model_terms(monkeypatch):
    rows = [{"en": f"term number {i}", "target": f"terim {i}"} for i in range(40)]
    monkeypatch.setattr(llm, "chat", Chat({"terms": rows}))
    text = "This survey covers attacks [1]. Earlier surveys [2] and a proof-of-concept [3] exist. **A Survey Title**"
    glossary, _ = writer.build_glossary([text, text], {}, "x", "tr", "m")
    assert sum(k.startswith("term number") for k in glossary) == writer.GLOSSARY_LIMIT
    assert glossary["survey"] == "derleme" and glossary["surveys"] == "derlemeler"
    assert glossary["proof-of-concept"] == "kavram kanıtı"
    assert "survey" not in writer.build_glossary(["**A Survey Title** [1] is cited."], {}, "x", "tr", "m")[0]


def test_quoted_titles_are_protected_and_left_out_of_the_measurement():
    text = "Bu argüman, “Unconditionally secure quantum bit commitment is impossible” adlı çalışmaya dayanır [16]."
    title = "Unconditionally secure quantum bit commitment is impossible"
    assert writer.protected_titles(text, [title]) == {title.lower()}
    assert language.term_audit(text, {"quantum": "kuantum"})["english_left"] == 0


def test_stray_brackets_are_removed_from_the_title():
    assert (
        writer.clean_title("Kuantum Kriptografi: Literatür İncelemesi】", "konu")
        == "Kuantum Kriptografi: Literatür İncelemesi"
    )
    assert writer.clean_title("  ", "konu") == "konu"
