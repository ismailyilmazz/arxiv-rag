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
