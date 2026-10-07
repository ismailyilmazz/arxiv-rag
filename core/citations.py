import re

MIN_PARAGRAPH_WORDS = 25
_CITE = re.compile(r"\[(\d+(?:\s*[,\u2013\-]\s*\d+)*)\]")
_REFERENCES_HEADING = re.compile(r"^#{1,3}\s*(References|Bibliography|Kaynakça|Kaynaklar)\s*$",
                                 re.IGNORECASE | re.MULTILINE)


def numbers_in(group: str) -> list[int]:
    out = []
    for part in re.split(r"\s*,\s*", group):
        bounds = re.split(r"\s*[\u2013\-]\s*", part)
        if len(bounds) == 2:
            lo, hi = int(bounds[0]), int(bounds[1])
            out.extend(range(lo, hi + 1) if lo <= hi else [lo, hi])
        else:
            out.append(int(bounds[0]))
    return out


def check_and_fix(markdown: str, n_sources: int, allowed_uncited: tuple = ()) -> tuple[str, dict]:
    heading = _REFERENCES_HEADING.search(markdown)
    removed_references = heading is not None
    if heading:
        markdown = markdown[:heading.start()].rstrip()
    invalid, cited = [], set()

    def fix(match):
        nums = numbers_in(match.group(1))
        good = [n for n in nums if 1 <= n <= n_sources]
        invalid.extend(n for n in nums if not 1 <= n <= n_sources)
        cited.update(good)
        return f"[{', '.join(str(n) for n in dict.fromkeys(good))}]" if good else ""

    fixed = _CITE.sub(fix, markdown)
    substantive, uncited, context, previous, exempt = 0, 0, False, "", False
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*] )", fixed):
        stripped = block.strip()
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            if level == 2:
                exempt = stripped[2:].strip() in allowed_uncited
            context = level >= 3 and bool(_CITE.search(stripped))
            previous = ""
            continue
        if exempt:
            continue
        if len(stripped.split()) >= MIN_PARAGRAPH_WORDS:
            substantive += 1
            lead_in = previous.endswith(":") and bool(_CITE.search(previous))
            if not (_CITE.search(stripped) or context or lead_in):
                uncited += 1
        if not stripped.startswith(("-", "*")):
            previous = stripped
    report = {
        "citations_found": len(_CITE.findall(markdown)),
        "invalid_numbers": invalid,
        "uncited_sources": sorted(set(range(1, n_sources + 1)) - cited),
        "paragraphs": substantive,
        "uncited_paragraphs": uncited,
        "removed_model_references": removed_references,
    }
    return fixed, report


def format_authors(raw) -> str:
    names = [n.strip() for n in re.split(r",|\band\b", raw or "") if n.strip()]
    if not names:
        return "Unknown"
    return ", ".join(names[:3]) + (" et al." if len(names) > 3 else "")


def bibliography(sources: list[dict], lang: str) -> str:
    heading = "## Kaynakça" if lang == "tr" else "## References"
    lines = [heading, ""]
    for s in sources:
        year = (s.get("published") or "")[:4] or "n.d."
        lines.append(f"[{s['n']}] {format_authors(s.get('authors'))} ({year}). {s['title']}. "
                     f"arXiv:{s['id']}. https://arxiv.org/abs/{s['id']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
