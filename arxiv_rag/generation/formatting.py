import re

from arxiv_rag.generation import language

META_NOTE = re.compile(
    r"\s*\((?=[^()]*(?:çevir|translat))(?=[^()]*(?:eksik|tamamlan|yapılamad|incomplete|missing|could not))[^()]{5,300}\)"
)


_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


_TERMINAL = re.compile(r"[.!?\])\"'\u201d]\s*$")


def _cells(line: str) -> tuple:
    return tuple(c.strip().lower() for c in line[line.index("|") :].strip().strip("|").split("|"))


def _trim_table_tail(lines: list[str]) -> tuple[list[str], bool]:
    changed = False
    while lines and lines[-1].lstrip().startswith("|"):
        start = len(lines) - 1
        while start > 0 and lines[start - 1].lstrip().startswith("|"):
            start -= 1
        run, tail = lines[start:], lines[-1].strip()
        if len(run) > 2 and not _SEPARATOR.match(tail) and tail.endswith("|"):
            cells = _cells(tail)
            if len(cells) == len(_cells(run[0])) and all(cells):
                break
        lines, changed = (lines[:-1] if len(run) > 2 else lines[:start]), True
        while lines and not lines[-1].strip():
            lines = lines[:-1]
    return lines, changed


def drop_repeated_heading(text: str, names) -> tuple[str, bool]:
    lines = text.lstrip().splitlines()
    wanted = {language.normalize(n) for n in names if n}
    if not lines or not wanted:
        return text, False
    core = lines[0].strip().lstrip("#").strip().strip("*_").strip().rstrip(":.").strip()
    if language.normalize(core) not in wanted:
        return text, False
    return "\n".join(lines[1:]).lstrip(), True


WIDE_CITATION = re.compile(
    r"[\u3010\uff3b]\s*(\d+(?:\s*[,\uff0c\u3001]\s*\d+)*)\s*(?:\u2020[^\u3011\uff3d]*)?[\u3011\uff3d]"
)


def normalize_citations(text: str) -> str:
    return WIDE_CITATION.sub(lambda m: "[" + re.sub(r"\s*[,\uff0c\u3001]\s*", ", ", m.group(1)) + "]", text)


def tidy_section(text: str, truncated: bool = False) -> tuple[str, bool]:
    text = normalize_citations(text)
    lines, seen, cut = text.rstrip().splitlines(), set(), None
    for i in range(1, len(lines)):
        if _SEPARATOR.match(lines[i]) and "|" in lines[i - 1]:
            header = _cells(lines[i - 1])
            if header in seen:
                cut = i - 1
                break
            seen.add(header)
    changed = cut is not None
    if changed:
        head = lines[cut][: lines[cut].index("|")].rstrip()
        lines = lines[:cut] + ([head] if head else [])
    if truncated:
        lines, trimmed = _trim_table_tail(lines)
        while lines and lines[-1].lstrip().startswith("#"):
            lines, trimmed = lines[:-1], True
        changed = changed or trimmed
    body = "\n".join(lines).rstrip()
    last = body.splitlines()[-1].strip() if body else ""
    kept = ("|",) if truncated else ("|", "-", "*", "#")
    if (
        last
        and not last.startswith(kept)
        and (truncated or not re.match(r"\d+\.", last))
        and not _TERMINAL.search(last)
    ):
        ends = [m.end() for m in re.finditer(r"[.!?](\s*\[[\d,\s\u2013-]+\])?[.]?(?=\s|$)", body)]
        if ends:
            body, changed = body[: ends[-1]].rstrip(), True
    return body, changed


def strip_meta_notes(text: str) -> str:
    return META_NOTE.sub("", text)


def _citation_numbers(text: str) -> set[int]:
    return {int(n) for group in re.findall(r"\[([\d,\s\u2013-]+)\]", text) for n in re.findall(r"\d+", group)}


CITATION = re.compile(r"\[(\d+(?:\s*[,\u2013\-]\s*\d+)*)\]")


def citation_numbers(group: str) -> list[int]:
    out = []
    for part in re.split(r"\s*,\s*", group):
        bounds = re.split(r"\s*[\u2013\-]\s*", part)
        if (
            len(bounds) == 2
            and all(b.isdigit() for b in bounds)
            and int(bounds[0]) <= int(bounds[1]) <= int(bounds[0]) + 50
        ):
            out.extend(range(int(bounds[0]), int(bounds[1]) + 1))
        elif bounds[0].isdigit():
            out.append(int(bounds[0]))
    return out


def ground_citations(text: str, allowed) -> tuple[str, int]:
    allowed, removed = set(allowed), 0

    def keep(match):
        nonlocal removed
        numbers = citation_numbers(match.group(1))
        kept = [n for n in numbers if n in allowed]
        removed += len(numbers) - len(kept)
        return f"[{', '.join(map(str, kept))}]" if kept else ""

    lines = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|") and not _SEPARATOR.match(line):
            first = CITATION.findall(line.strip().strip("|").split("|")[0])
            numbers = [n for g in first for n in citation_numbers(g)]
            if numbers and not any(n in allowed for n in numbers):
                removed += len(numbers)
                continue
        new = CITATION.sub(keep, line)
        if new != line:
            new = re.sub(r"[ \t]+([.,;:!?)])", r"\1", new)
            new = re.sub(r"(?<=\S) {2,}(?=\S)", " ", new)
        lines.append(new)
    return "\n".join(lines), removed


def _row_key(line: str) -> tuple:
    first = _cells(line)[0]
    cited = CITATION.findall(first)
    return ("cite", tuple(citation_numbers(cited[0]))) if cited else ("text", " ".join(first.split()))


def _repair_table(block: list[str]) -> list[str]:
    if len(block) < 2 or not _SEPARATOR.match(block[1]):
        return block
    width, head, rows, seen = len(_cells(block[0])), _row_key(block[0]), [], set()
    for line in block[2:]:
        if _SEPARATOR.match(line) or len(_cells(line)) != width:
            continue
        key = _row_key(line)
        if key == head or key in seen:
            continue
        seen.add(key)
        rows.append(line)
    return [block[0], block[1]] + rows if rows else []


def repair_tables(text: str) -> tuple[str, bool]:
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        out.extend(_repair_table(lines[i:j]))
        i = j
    fixed = "\n".join(out)
    return fixed, fixed != text


_LIST_START = re.compile(r"^\s*(?:[-+>|#]|\*\s|\d+[.)]\s)")


def _label_line(line: str) -> bool:
    text = line.strip()
    return (
        0 < len(text.split()) <= 8
        and text[0].isalpha()
        and not _LIST_START.match(line)
        and not CITATION.search(text)
        and text[-1] not in ".!?:;,*"
    )


def inline_label_lines(text: str) -> str:
    blocks = re.split(r"(\n\s*\n)", text)
    for k, block in enumerate(blocks):
        lines = block.split("\n")
        if len(lines) >= 2 and _label_line(lines[0]) and lines[1].strip() and not _LIST_START.match(lines[1]):
            blocks[k] = "\n".join([f"{lines[0].strip()}. {lines[1].lstrip()}"] + lines[2:])
    return "".join(blocks)
