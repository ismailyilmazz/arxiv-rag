from arxiv_rag.generation import citations, formatting, writer

HEADER = "| Work | Year | Key idea | Setting / scale | Main result |\n|------|------|----------|-----------------|------------|"
ROWS = [
    "| [1] Perspectives | 2025 | Survey | Reports | Risks of over-reliance |",
    "| [2] Attitudes | 2025 | Survey | 117 students | Two-thirds use AI weekly |",
    "| [3] Artificial or Human Intelligence? | 2025 | Model | Theory | Threshold T |",
    "| [4] Computer Vision | 2025 | Course | 12 students | Self-efficacy gains |",
    "| [5] Age of Generative AI | 2025 | Framework | Conceptual | Originality concerns |",
]
BROKEN = "\n".join(
    [
        HEADER,
        *ROWS,
        "| [21] AI Meets the Classroom | 2024 | Lab experiments | Work | Year | Key idea | Setting or scale | Main result |",
    ]
    + ["|---|---|---|---|---|", *[r.replace("Perspectives", "Perspectives and potential issues") for r in ROWS]]
    + [
        "| [21] AI Meets the Classroom | 2024 | Lab experiments | Undergraduates | No overall effect |",
        "| [22] The GPT Surprise | 2024 | Randomized trial | 5,831 students | Adopters scored higher |",
        "| [15] Ensuring CS Learning | 2026 | Written quizzes | Programming courses | Quizzes verify mastery |",
    ]
)


def test_a_table_restarted_by_completion_keeps_one_header_and_unique_rows():
    fixed, changed = formatting.repair_tables(
        "Comparison Table\n\n" + BROKEN + "\n\nThe literature reveals a tension [22]."
    )
    lines = [line for line in fixed.splitlines() if line.startswith("|")]
    firsts = [formatting.CITATION.findall(line.split("|")[1]) for line in lines[2:]]
    assert changed and lines[0] == HEADER.splitlines()[0]
    assert sum(bool(formatting._SEPARATOR.match(line)) for line in lines) == 1
    assert firsts == [["1"], ["2"], ["3"], ["4"], ["5"], ["21"], ["22"], ["15"]]
    assert fixed.endswith("The literature reveals a tension [22].")
    assert formatting.repair_tables("| a | b |\n|---|---|\n| [1] x | y |") == (
        "| a | b |\n|---|---|\n| [1] x | y |",
        False,
    )


def test_citations_outside_what_a_section_was_shown_are_removed():
    text = "Adoption is high [2, 4, 7]. Integrity matters [1, 5].\n\n| Work | Year |\n|---|---|\n| [7] Other | 2023 |\n| [4] Vision | 2025 |"
    out, removed = formatting.ground_citations(text, {1, 4})
    assert out.startswith("Adoption is high [4]. Integrity matters [1].")
    assert "[7] Other" not in out and "| [4] Vision | 2025 |" in out and removed == 4
    assert formatting.ground_citations("Conclusion text [3, 6]. More [2-3] here.", set()) == (
        "Conclusion text. More here.",
        4,
    )


def test_each_section_may_cite_only_the_sources_it_was_shown():
    sources = [
        {"id": "a", "n": 1, "card": {"problem": "p"}},
        {"id": "b", "n": 2},
        {"id": "c", "n": 3},
        {"id": "d", "n": 4},
    ]
    themes = [{"ids": ["a", "b"]}, {"ids": ["c", "d"]}]
    assert writer.allowed_numbers("theme", sources, themes, themes[1]) == {3, 4}
    assert writer.allowed_numbers("brief", sources, themes) == writer.allowed_numbers("limits", sources, themes) == {1}
    assert writer.allowed_numbers("setup", sources, themes) == {1, 2, 3, 4}
    assert writer.allowed_numbers("written", sources, themes) == set()
    assert [s["n"] for s in writer.setup_others(sources, themes)] == [2, 3, 4]


def test_label_lines_join_their_paragraph_before_translation():
    text = (
        "Theoretical incentive analysis  \n*Artificial or Human Intelligence?* formalizes the shift [3].\n\n"
        "- first item\n- second item\n\nComparison Table\n\n| a | b |\n|---|---|"
    )
    out = formatting.inline_label_lines(text)
    assert out.startswith(
        "Theoretical incentive analysis. *Artificial or Human Intelligence?* formalizes the shift [3]."
    )
    assert "- first item\n- second item" in out and "Comparison Table\n\n| a | b |" in out
    sentence = "The results are strong.\nThey hold across settings [2]."
    assert formatting.inline_label_lines(sentence) == sentence


def test_number_audit_treats_turkish_thousands_as_the_same_number():
    report = citations.number_audit("Çalışma 5.831 öğrenenle yapıldı [22].", {22: "a class of 5,831 learners"})
    assert report["checked"] == 1 and report["unsupported"] == 0
