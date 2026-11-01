import io

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE

from arxiv_rag.generation.docx_export import to_docx

SAMPLE = """# Detecting Hallucinations: A Survey

## Introduction

Large models **often hallucinate** facts [1, 2]. See *recent* work and `code` too.
This line continues the same paragraph.

- First point [3]
- Second point

1. Step one
2. Step two

| Work | Year | Result |
|------|------|--------|
| [1] Paper A | 2025 | **82 %** accuracy |
| [2] Paper B | 2023 |

> A quoted remark.

---

### Büyük dil modelleri

Türkçe paragraf **kalın** ve _italik_ içerir; snake_case_name olduğu gibi kalır.

## References

[1] A. Author (2025). Title one. arXiv:2512.22416. https://arxiv.org/abs/2512.22416

[2] B. Author (2023). Title two. arXiv:2311.13230. https://arxiv.org/abs/2311.13230
"""


def load(markdown=SAMPLE):
    return Document(io.BytesIO(to_docx(markdown)))


def test_headings_paragraphs_and_inline_styles_become_word_structures():
    doc = load()
    styled = [(p.style.name, p.text) for p in doc.paragraphs if p.text]
    assert styled[0] == ("Title", "Detecting Hallucinations: A Survey")
    assert doc.core_properties.title == "Detecting Hallucinations: A Survey"
    assert ("Heading 1", "Introduction") in styled and ("Heading 2", "Büyük dil modelleri") in styled
    para = next(p for p in doc.paragraphs if p.text.startswith("Large models"))
    assert para.text.endswith("This line continues the same paragraph.") and "[1, 2]" in para.text
    assert [r.text for r in para.runs if r.bold] == ["often hallucinate"]
    assert [r.text for r in para.runs if r.italic] == ["recent"]
    assert [r.text for r in para.runs if r.font.name == "Consolas"] == ["code"]
    turkish = next(p for p in doc.paragraphs if p.text.startswith("Türkçe"))
    assert [r.text for r in turkish.runs if r.italic] == ["italik"] and "snake_case_name" in turkish.text


def test_lists_and_quotes_use_word_styles():
    doc = load()
    assert [p.text for p in doc.paragraphs if p.style.name == "List Bullet"] == ["First point [3]", "Second point"]
    assert [p.text for p in doc.paragraphs if p.style.name == "List Number"] == ["Step one", "Step two"]
    assert [p.text for p in doc.paragraphs if p.style.name == "Quote"] == ["A quoted remark."]


def test_tables_keep_their_shape_and_bold_header():
    table = load().tables[0]
    rows = [[cell.text for cell in row.cells] for row in table.rows]
    assert rows == [["Work", "Year", "Result"], ["[1] Paper A", "2025", "82 % accuracy"], ["[2] Paper B", "2023", ""]]
    assert all(run.bold for run in table.rows[0].cells[0].paragraphs[0].runs)
    assert [r.text for r in table.rows[1].cells[2].paragraphs[0].runs if r.bold] == ["82 %"]


def test_references_stay_separate_and_their_addresses_are_clickable():
    doc = load()
    refs = [p.text for p in doc.paragraphs if p.text.startswith("[")]
    assert refs[0].startswith("[1] A. Author") and refs[1].startswith("[2] B. Author") and len(refs) == 2
    links = {r.target_ref for r in doc.part.rels.values() if r.reltype == RELATIONSHIP_TYPE.HYPERLINK}
    assert links == {"https://arxiv.org/abs/2512.22416", "https://arxiv.org/abs/2311.13230"}


def test_markdown_links_and_documents_without_a_title():
    doc = load("Plain text with a [label](https://example.org/x) link.")
    assert doc.paragraphs[0].text == "Plain text with a label link."
    assert {r.target_ref for r in doc.part.rels.values() if r.reltype == RELATIONSHIP_TYPE.HYPERLINK} == {
        "https://example.org/x"
    }
