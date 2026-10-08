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


def _entry(s: dict) -> str:
    year = (s.get("published") or "")[:4] or "n.d."
    return (f"[{s['n']}] {format_authors(s.get('authors'))} ({year}). {s['title']}. "
            f"arXiv:{s['id']}. https://arxiv.org/abs/{s['id']}")


def bibliography(sources: list[dict], lang: str, context: list[dict] = ()) -> str:
    heading = "## Kaynakça" if lang == "tr" else "## References"
    lines = [heading, ""]
    if context:
        lines += ["### İncelenen çalışmalar" if lang == "tr" else "### Reviewed works", ""]
    for s in sources:
        lines += [_entry(s), ""]
    if context:
        lines += ["### Bağlam çalışmaları (yalnızca özet)" if lang == "tr" else "### Context works (summary only)", ""]
        for c in context:
            lines += [_entry(c), ""]
    return "\n".join(lines).rstrip() + "\n"


_META = re.compile(r"\b(summary|summaries|reading card|the card|the material|context source|contextual source|"
                   r"özet(?:i|inde)? kart|bağlam kayna\w*)\b", re.IGNORECASE)


def renumber(markdown: str) -> tuple[str, dict[int, int]]:
    mapping: dict[int, int] = {}

    def sub(match):
        nums = numbers_in(match.group(1))
        for n in nums:
            mapping.setdefault(n, len(mapping) + 1)
        return "[" + ", ".join(str(m) for m in sorted(dict.fromkeys(mapping[n] for n in nums))) + "]"

    return _CITE.sub(sub, markdown), mapping


def aliases(title: str) -> set[str]:
    out = set()
    head = re.split(r"[:\u2014\u2013]", title)[0].strip()
    if 1 <= len(head.split()) <= 3:
        out.add(head)
    for token in re.findall(r"[A-Za-z][\w-]*", title):
        if len(token) >= 3 and (token.isupper() or re.search(r"[a-z][A-Z]", token)):
            out.add(token)
    return out


def name_mismatches(markdown: str, sources: list[dict]) -> dict:
    names = {s["n"]: aliases(s["title"]) for s in sources}
    counts: dict[str, int] = {}
    for alias_set in names.values():
        for a in alias_set:
            counts[a] = counts.get(a, 0) + 1
    unique = {n: {a for a in al if counts[a] == 1} for n, al in names.items()}
    examples, total = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", markdown):
        cited = {m for g in _CITE.findall(sentence) for m in numbers_in(g)}
        if not cited:
            continue
        for n, al in unique.items():
            if n not in cited and any(re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", sentence) for a in al):
                total += 1
                if len(examples) < 5:
                    examples.append(" ".join(sentence.split())[:200])
    return {"count": total, "examples": examples}


def meta_language(markdown: str) -> dict:
    found = [m.group(0) for m in _META.finditer(markdown)]
    return {"count": len(found), "examples": found[:5]}


def further_reading(sources: list[dict], lang: str) -> str:
    if not sources:
        return ""
    lines = ["## İleri okuma" if lang == "tr" else "## Further reading", ""]
    for s in sources:
        year = (s.get("published") or "")[:4] or "n.d."
        lines.append(f"- {s['title']} ({year}). https://arxiv.org/abs/{s['id']}")
    return "\n".join(lines) + "\n"


_NUM_LOOSE = re.compile(r"\d+(?:[.,]\d+)*")
_NUM_STRICT = re.compile(r"(?<![A-Za-z_\-\d.,])\d+(?:[.,]\d+)*")


def _norm_number(raw: str):
    if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):
        raw = raw.replace(",", "")
    raw = raw.replace(",", ".")
    if re.fullmatch(r"(19|20)\d\d", raw):
        return None
    return raw.rstrip("0").rstrip(".") if "." in raw else raw


def _numbers(text: str, pattern) -> set[str]:
    return {v for v in (_norm_number(m) for m in pattern.findall(text)) if v}


def number_audit(markdown: str, materials: dict[int, str]) -> dict:
    checked, missing, examples = 0, 0, []
    for sentence in re.split(r"(?<=[.!?])\s+", markdown):
        if sentence.lstrip().startswith(("|", "#")):
            continue
        cited = {n for g in _CITE.findall(sentence) for n in numbers_in(g)}
        if len(cited) != 1 or next(iter(cited)) not in materials:
            continue
        n = next(iter(cited))
        plain = _CITE.sub(" ", sentence)
        have = _numbers(materials[n], _NUM_LOOSE)
        for value in _numbers(plain, _NUM_STRICT):
            checked += 1
            if value not in have:
                missing += 1
                if len(examples) < 5:
                    examples.append(f"[{n}] {value}: {' '.join(plain.split())[:160]}")
    return {"checked": checked, "unsupported": missing, "examples": examples}
