import io
import re

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
QUOTE = re.compile(r"^\s*>\s?(.*)$")
RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
REFERENCE = re.compile(r"^\[\d+\]\s")
INLINE = re.compile(
    r"(\*\*[^*]+\*\*"
    r"|(?<![\w*])\*[^*\s][^*]*\*(?![\w*])"
    r"|(?<!\w)_[^_\s][^_]*_(?!\w)"
    r"|`[^`]+`"
    r"|\[[^\]]+\]\(https?://[^)\s]+\)"
    r"|https?://[^\s<>()]+[^\s<>().,;:!?\"'])"
)


def _hyperlink(paragraph, url: str, label: str) -> None:
    rel = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(color)
    props.append(underline)
    run.append(props)
    text = OxmlElement("w:t")
    text.set(qn("xml:space"), "preserve")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def add_inline(paragraph, text: str) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            paragraph.add_run(part[1:-1]).font.name = "Consolas"
        elif part.startswith("[") and "](" in part and part.endswith(")"):
            label, url = part[1:-1].split("](", 1)
            _hyperlink(paragraph, url, label)
        elif part.startswith("http"):
            _hyperlink(paragraph, part, part)
        elif len(part) > 2 and part[0] == part[-1] and part[0] in "*_":
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table(document, rows: list[str]) -> None:
    body = [_cells(row) for row in rows if not SEPARATOR.match(row)]
    width = max(len(row) for row in body)
    table = document.add_table(rows=len(body), cols=width)
    table.style = "Table Grid"
    for r, row in enumerate(body):
        for c in range(width):
            paragraph = table.cell(r, c).paragraphs[0]
            add_inline(paragraph, row[c] if c < len(row) else "")
            if r == 0:
                for run in paragraph.runs:
                    run.bold = True


def to_docx(markdown: str) -> bytes:
    document = Document()
    lines = markdown.replace("\r\n", "\n").split("\n")
    paragraph, i = [], 0

    def flush():
        if paragraph:
            add_inline(document.add_paragraph(), " ".join(line.strip() for line in paragraph))
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        heading = HEADING.match(line)
        if not line.strip() or RULE.match(line):
            flush()
        elif heading:
            flush()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            if level == 1 and not document.core_properties.title:
                document.core_properties.title = text
            add_inline(document.add_heading(level=0 if level == 1 else min(level - 1, 4)), text)
        elif line.lstrip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            _table(document, rows)
            continue
        elif BULLET.match(line) and not RULE.match(line):
            flush()
            add_inline(document.add_paragraph(style="List Bullet"), BULLET.match(line).group(1))
        elif NUMBERED.match(line):
            flush()
            add_inline(document.add_paragraph(style="List Number"), NUMBERED.match(line).group(1))
        elif QUOTE.match(line):
            flush()
            add_inline(document.add_paragraph(style="Quote"), QUOTE.match(line).group(1))
        elif REFERENCE.match(line):
            flush()
            add_inline(document.add_paragraph(), line.strip())
        else:
            paragraph.append(line)
        i += 1
    flush()
    out = io.BytesIO()
    document.save(out)
    return out.getvalue()
