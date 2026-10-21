import re

from arxiv_rag.generation import gemini, language, llm

LANGUAGES = {"en": "English", "tr": "Turkish"}
STYLE = (
    "Write complete sentences, each with an explicit subject and a finite verb; in Turkish every sentence ends "
    "with its predicate. Prefer flowing paragraphs; use bullet points only for short enumerations and write "
    "each bullet as a full sentence."
)
TOKEN_BUDGET = 7600


SECTION_PROMPT = """You are writing one section of a {label} in {language}, based only on numbered sources.

Document outline:
{outline}

Already written:
{written}

Now write: {target}
Task: {task}

Material:
{material}

Rules:
- {style}
- Every paragraph that relies on the sources cites them in square brackets, such as [1] or [1, 3], using only the numbers 1 to {n}. When you mention a source by name, put its number next to it.
- Do not invent results, numbers, data sets or methods that are not in the material.
- Do not repeat what is already written. Do not write the section heading, any other section or a reference list.
- Length: as long as the material supports for this part, and no longer. Never pad.{extra}
"""

TITLE_PROMPT = """Write a concise, specific title in {language} for a {label} with this content:
{written}
Return only the title text."""


def _year(s: dict) -> str:
    return (s.get("published") or "")[:4] or "n.d."


def card_text(s: dict, keys=("problem", "method", "setup", "findings", "limitations"), limit: int = 0) -> str:
    lines = [f"[{s['n']}] {s['title']} ({_year(s)})"]
    lines += [f"{k.capitalize()}: {s['card'][k][:limit] if limit else s['card'][k]}" for k in keys if s["card"].get(k)]
    return "\n".join(lines)


def summary_of(written: list[tuple[str, str]]) -> str:
    if not written:
        return "Nothing yet."
    lines = []
    for heading, text in written:
        firsts = [
            re.split(r"(?<=[.!?])\s", p.strip(), maxsplit=1)[0]
            for p in re.split(r"\n\s*\n", text)
            if p.strip() and not p.lstrip().startswith("#")
        ]
        lines.append(f"{heading}: " + " ".join(firsts)[:400])
    return "\n".join(lines)


CONTINUE = (
    "\n\nThe section you are writing currently ends with:\n{tail}\n\nContinue exactly from where it stops. "
    "Do not repeat earlier text, do not start a new table and do not repeat headings. Finish the section "
    "with complete sentences."
)

TRANSLATE_PROMPT = """Translate the following section of a {label} from English into {language}.
- Keep every citation such as [3] or [1, 4] exactly as it is and where it is.
- Paper titles are never translated: keep them exactly as they are wherever they appear (in bold, in italics, in
  quotation marks, in the Work column of a table). Keep model names and acronyms in English too.{titles}
- Keep the markdown structure (table rows and columns, bold text, lists) and translate everything else, including
  the table header and the other table cells.
- Translate other technical terms into the {language} terms of the field.{glossary}
- Write natural, fluent {language}; in Turkish every sentence ends with its predicate.
- Do not add, remove or summarize content. Return only the translated section.

Section:
{text}"""

RETRY_TRANSLATE_PROMPT = """The previous translation attempt returned English. Translate the text below into {language}.
- Every sentence, heading, bold or italic phrase and table cell of your output must be in {language}, except paper
  titles: keep paper titles exactly as they are (in bold, in italics, in quotation marks, in the Work column of a
  table). Model names and acronyms may stay in English.{titles}{glossary}
- Keep every citation such as [3] or [1, 4] exactly as it is and where it is, and keep the markdown structure.
- Do not add notes or comments about the text. Return only the translation.

Text:
{text}"""

HEADINGS_PROMPT = """Translate each line below into {language}. Keep model names and acronyms in English.
Return exactly {n} lines in the same order and nothing else.

{lines}"""

HEADING_RETRY_PROMPT = """Translate this heading into {language}: {line}
The result must be a {language} phrase. Translate technical words too (for example, a Turkish heading says dolanıklık,
not entanglement). Only acronyms (such as QKD or BB84) and model names may stay in English.
Return only the translated heading."""

BASE_TERMS = {
    "tr": {
        "survey": "derleme",
        "surveys": "derlemeler",
        "proof-of-concept": "kavram kanıtı",
    }
}
GLOSSARY_LIMIT = 30
TABLE_HEADERS = {
    "tr": {
        "work": "Çalışma",
        "year": "Yıl",
        "key idea": "Ana fikir",
        "setting or scale": "Ortam ya da ölçek",
        "setting / scale": "Ortam / ölçek",
        "setting": "Ortam",
        "main result": "Ana sonuç",
        "main finding": "Ana bulgu",
    }
}

GLOSSARY_PROMPT = """A {label} will be translated from English into {language} section by section. These words and phrases
occur often in it:
{candidates}

Choose up to 30 of them that are technical terms of the field and give the one {language} term that every section
must use for each. Use the established {language} term of the field. Leave out acronyms, model names, method names
and paper titles.
Return JSON: {{"terms": [{{"en": "...", "target": "..."}}]}}"""

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


def titles_rule(text: str, titles=()) -> str:
    found = sorted(protected_titles(text, titles))
    named = [t for t in titles if language.normalize(t) in found]
    if not named:
        return "\n- This text names no paper titles, so translate every bold, italic or quoted phrase."
    listed = "; ".join(f"\u201c{t}\u201d" for t in dict.fromkeys(named))
    return (
        f"\n- The paper titles in this text are: {listed}. Keep exactly these unchanged; every other bold, italic or "
        "quoted phrase is not a title and must be translated."
    )


def english_labels(text: str, titles=()) -> list[str]:
    found = protected_titles(text, titles)
    labels = []
    for match in language.SPAN.finditer(text):
        span = next(g for g in match.groups() if g)
        norm = language.normalize(span)
        if norm in found or any(t in norm for t in found if len(t) >= 25):
            continue
        if language.text_is_english(span):
            labels.append(span.strip())
    return labels


def localize_labels(text: str, names: dict, titles=()) -> str:
    known = {language.normalize(t) for t in titles}
    local = {
        language.normalize(en): target
        for en, target in names.items()
        if en and target and not same_text(en, target) and language.normalize(en) not in known
    }
    if not local:
        return text

    def swap(match):
        span = next(g for g in match.groups() if g)
        target = local.get(language.normalize(span))
        return match.group(0).replace(span, target) if target else match.group(0)

    return language.SPAN.sub(swap, text)


CITED_AFTER = re.compile(r"[\s,;:.)\u2019']*(\(\d{4}\)\s*)?\[\d")


def unbold_labels(text: str, titles=()) -> str:
    keep = protected_titles(text, titles)

    def plain(match):
        if match.group(0)[0] != "*":
            return match.group(0)
        span = next(g for g in match.groups() if g)
        norm = language.normalize(span)
        if norm in keep or any(t in norm for t in keep if len(t) >= 25):
            return match.group(0)
        return span

    return language.SPAN.sub(plain, text)


def protected_titles(source: str, titles=()) -> set[str]:
    known = {language.normalize(t) for t in titles if t and t.strip()}
    found = set()
    for match in language.SPAN.finditer(source):
        span = language.normalize(next(g for g in match.groups() if g))
        cited = bool(CITED_AFTER.match(source[match.end() : match.end() + 24]))
        if (span in known or (len(span) >= 20 and any(span in t for t in known))) and (len(span) >= 25 or cited):
            found.add(span)
        else:
            found.update(t for t in known if len(t) >= 25 and t in span)
    return found


def is_complete(candidate: str, source: str, min_ratio: float = 0.7, titles=()) -> bool:
    kept = language.normalize(candidate)
    return (
        _citation_numbers(source) <= _citation_numbers(candidate)
        and len(candidate.strip()) >= min_ratio * len(source.strip())
        and all(t in kept for t in protected_titles(source, titles))
    )


def _cells_of(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_runs(lines: list[str]) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, line in enumerate(lines + [""]):
        if line.lstrip().startswith("|"):
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i))
            start = None
    return runs


def localize_table_headers(text: str, lang: str) -> str:
    names = TABLE_HEADERS.get(lang)
    if not names or "|" not in text:
        return text
    lines = text.splitlines()
    for start, _ in _table_runs(lines):
        cells = _cells_of(lines[start])
        lines[start] = "| " + " | ".join(names.get(" ".join(c.lower().split()), c) for c in cells) + " |"
    return "\n".join(lines)


def fix_tables(translated: str, source: str, lang: str) -> str:
    return localize_table_headers(keep_work_column(translated, source), lang)


def keep_work_column(translated: str, source: str) -> str:
    if "|" not in source:
        return translated
    src, out = source.splitlines(), translated.splitlines()
    src_runs, out_runs = _table_runs(src), _table_runs(out)
    if len(src_runs) != len(out_runs):
        return translated
    for (a, b), (c, d) in zip(src_runs, out_runs):
        if b - a != d - c or _cells_of(src[a])[0].lower() != "work":
            continue
        for i in range(1, b - a):
            if language.SEPARATOR.match(out[c + i]):
                continue
            src_cells, out_cells = _cells_of(src[a + i]), _cells_of(out[c + i])
            if len(src_cells) == len(out_cells):
                out[c + i] = "| " + " | ".join([src_cells[0]] + out_cells[1:]) + " |"
    return "\n".join(out)


def glossary_rule(glossary) -> str:
    if not glossary:
        return ""
    pairs = "; ".join(f"{en} = {target}" for en, target in glossary.items())
    return f"\n- Outside paper titles, always use these translations of terms and theme names: {pairs}."


REPAIR_TOKENS = 2000
GEMINI_REPAIR_TOKENS = 8000


def new_repair_stats() -> dict:
    return {"units": 0, "gemini_tried": 0, "gemini_ok": 0, "groq_tried": 0, "groq_ok": 0, "kept": 0}


def _repair_engines() -> tuple:
    return ("gemini", "groq") if gemini.available() and gemini.status().get("ok") else ("groq", "groq")


def _repair_call(engine: str, prompt: str, model: str) -> tuple[str, int]:
    if engine == "gemini":
        try:
            text, _, finish = gemini.complete(prompt, max_tokens=GEMINI_REPAIR_TOKENS)
        except gemini.GeminiUnavailable:
            return "", 0
        return ("" if finish == "length" else text), 0
    text, tokens, _ = _call(prompt, model, REPAIR_TOKENS)
    return text, tokens


BATCH_REPAIR_PROMPT = """The previous translation attempt left the numbered text units below in English. Translate
every unit into {language}.
- Every sentence, heading, bold or italic phrase and table cell must be in {language}, except paper titles: keep paper
  titles exactly as they are (in bold, in italics, in quotation marks, in the Work column of a table). Model names and
  acronyms may stay in English.{titles}{glossary}
- Keep every citation such as [3] or [1, 4] exactly as it is and where it is, and keep the markdown structure of each
  unit.
- Translate each unit on its own: do not merge, split or summarize units, and do not add notes.
Return JSON: {{"units": [{{"id": 1, "text": "..."}}]}}

{units}"""


def _gemini_batch(blocks: list[str], lang: str, glossary, titles) -> dict:
    units = "\n\n".join(f"Unit {k}:\n<<<\n{block}\n>>>" for k, block in enumerate(blocks, start=1))
    prompt = BATCH_REPAIR_PROMPT.format(
        language=LANGUAGES[lang],
        units=units,
        glossary=glossary_rule(glossary),
        titles=titles_rule("\n\n".join(blocks), titles),
    )
    try:
        data, _ = gemini.chat(prompt, max_tokens=GEMINI_REPAIR_TOKENS * 2)
    except gemini.GeminiUnavailable:
        return {}
    out = {}
    for row in data.get("units") or []:
        if isinstance(row, dict) and str(row.get("id", "")).isdigit() and isinstance(row.get("text"), str):
            out[int(row["id"])] = row["text"]
    return out


def _accepted(piece: str, block: str, titles) -> bool:
    return bool(
        piece
        and is_complete(piece, block, titles=titles)
        and not language.block_is_english(piece)
        and not english_labels(piece, titles)
    )


def repair_paragraphs(
    text: str, lang: str, model: str, tables: bool = False, glossary=None, titles=(), stats=None
) -> tuple[str, int, int, int]:
    stats = stats if stats is not None else new_repair_stats()
    blocks, tokens, repaired, failed = re.split(r"\n\s*\n", text.strip()), 0, 0, 0
    wanted = [
        i
        for i, block in enumerate(blocks)
        if (tables and block.lstrip().startswith("|"))
        or language.block_is_english(block)
        or english_labels(block, titles)
    ]
    engines = _repair_engines() if wanted else ()
    batch = _gemini_batch([blocks[i] for i in wanted], lang, glossary, titles) if "gemini" in engines else {}
    for k, i in enumerate(wanted, start=1):
        block = blocks[i]
        stats["units"] += 1
        prompt = RETRY_TRANSLATE_PROMPT.format(
            language=LANGUAGES[lang], text=block, glossary=glossary_rule(glossary), titles=titles_rule(block, titles)
        )
        for engine in engines:
            stats[f"{engine}_tried"] += 1
            if engine == "gemini":
                piece = batch.get(k, "")
            else:
                piece, extra = _repair_call(engine, prompt, model)
                tokens += extra
            piece = fix_tables(strip_meta_notes(piece.strip()), block, lang)
            if _accepted(piece, block, titles):
                blocks[i], repaired = piece, repaired + 1
                stats[f"{engine}_ok"] += 1
                break
        else:
            failed += 1
            stats["kept"] += 1
    return "\n\n".join(blocks), tokens, repaired, failed


def translate_section(
    text: str, label: str, lang: str, model: str, glossary=None, titles=(), stats=None
) -> tuple[str, int, bool, int]:
    out, tokens, truncated = _call(
        TRANSLATE_PROMPT.format(
            label=label,
            language=LANGUAGES[lang],
            text=text,
            glossary=glossary_rule(glossary),
            titles=titles_rule(text, titles),
        ),
        model,
        4000,
    )
    out = strip_meta_notes(out)
    if lang == "en":
        return out, tokens, truncated, 1
    out = fix_tables(out, text, lang)
    if truncated or not is_complete(out, text, titles=titles):
        out, extra, _, _ = repair_paragraphs(
            text, lang, model, tables=True, glossary=glossary, titles=titles, stats=stats
        )
        return out, tokens + extra, False, 2
    if language.has_english(out) or english_labels(out, titles):
        out, extra, repaired, _ = repair_paragraphs(out, lang, model, glossary=glossary, titles=titles, stats=stats)
        return out, tokens + extra, truncated, 2 if repaired else 1
    return out, tokens, truncated, 1


def _heading_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip("*").strip().strip('"').strip()
        if line and not line.endswith(":"):
            lines.append(line)
    return lines


def _ask_headings(lines: list[str], lang: str, model: str) -> tuple[list[str], int]:
    prompt = HEADINGS_PROMPT.format(language=LANGUAGES[lang], n=len(lines), lines="\n".join(lines))
    text, tokens, _ = llm.complete(prompt, model=model, max_tokens=800)
    return _heading_lines(text), tokens


def same_text(a: str, b: str) -> bool:
    return " ".join(a.translate(language.HYPHENS).lower().split()) == " ".join(
        b.translate(language.HYPHENS).lower().split()
    )


def plausible_heading(candidate: str, original: str) -> bool:
    return (
        not same_text(candidate, original)
        and len(candidate) <= 2 * len(original) + 20
        and not re.search(r"\[\d", candidate)
    )


def translate_lines(lines: list[str], lang: str, model: str) -> tuple[list[str], int]:
    if lang == "en" or not lines:
        return list(lines), 0
    got, tokens = _ask_headings(lines, lang, model)
    result = got if len(got) == len(lines) else list(lines)
    for i, line in enumerate(lines):
        if not same_text(result[i], line) and not language.english_words(result[i]):
            continue
        text, extra, _ = llm.complete(
            HEADING_RETRY_PROMPT.format(language=LANGUAGES[lang], line=line), model=model, max_tokens=400
        )
        tokens += extra
        single = _heading_lines(text)
        if len(single) == 1 and plausible_heading(single[0], line):
            if same_text(result[i], line) or len(language.english_words(single[0])) < len(
                language.english_words(result[i])
            ):
                result[i] = single[0]
    return result, tokens


def build_glossary(texts: list[str], names: dict, label: str, lang: str, model: str) -> tuple[dict, int]:
    glossary, tokens = {}, 0
    candidates = language.frequent_terms("\n\n".join(texts))
    if lang != "en" and candidates:
        prompt = GLOSSARY_PROMPT.format(
            label=label, language=LANGUAGES[lang], candidates="\n".join(f"- {c}" for c in candidates)
        )
        try:
            data, tokens = llm.chat(prompt, model=model, max_tokens=1500)
        except Exception:
            data = {}
        for row in data.get("terms") or []:
            if not isinstance(row, dict):
                continue
            en, target = str(row.get("en") or "").strip(), str(row.get("target") or "").strip()
            if en and target and len(en) <= 60 and len(target) <= 60 and not en.isupper():
                glossary[en] = target
            if len(glossary) >= GLOSSARY_LIMIT:
                break
    plain = language._translatable("\n\n".join(texts)).lower()
    glossary.update(
        {
            en: target
            for en, target in BASE_TERMS.get(lang, {}).items()
            if re.search(rf"(?<![a-z]){re.escape(en)}(?![a-z])", plain)
        }
    )
    glossary.update({en: local for en, local in names.items() if en and local and not same_text(en, local)})
    return glossary, tokens


def _stem(text: str) -> str:
    words = re.findall(r"[^\W\d_]+", text.lower())
    return words[0][:5] if words else ""


def misaligned(english: str, local: str, glossary: dict, skip=()) -> bool:
    name, lowered = language.normalize(english), local.lower()
    for key, target in glossary.items():
        term = language.normalize(key)
        if key in skip or term == name or len(term) < 4 or same_text(key, target):
            continue
        stem = _stem(target)
        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", name) and len(stem) >= 4 and stem not in lowered:
            return True
    return False


def align_names(names: dict, glossary: dict, lang: str, model: str) -> tuple[int, int]:
    tokens, fixed = 0, 0
    if lang == "en":
        return tokens, fixed
    terms = {k: v for k, v in glossary.items() if k not in names}
    for english, local in list(names.items()):
        if not misaligned(english, local, glossary, names):
            continue
        prompt = HEADING_RETRY_PROMPT.format(language=LANGUAGES[lang], line=english) + glossary_rule(terms)
        text, extra, _ = llm.complete(prompt, model=model, max_tokens=400)
        tokens += extra
        single = _heading_lines(text)
        if (
            len(single) == 1
            and plausible_heading(single[0], english)
            and not misaligned(english, single[0], glossary, names)
        ):
            names[english] = single[0]
            if english in glossary:
                glossary[english] = single[0]
            fixed += 1
    return tokens, fixed


def translate_sections(
    sections: list[dict], label: str, lang: str, model: str, names: dict, progress=None, titles=()
) -> tuple[dict, list[dict], int, dict]:
    progress = progress or (lambda stage, done, total: None)
    glossary, used = build_glossary([s["text"] for s in sections], names, label, lang, model)
    extra, fixed = align_names(names, glossary, lang, model)
    report = [
        {
            "section": "translate:glossary",
            "source": None,
            "tokens": used + extra,
            "truncated": False,
            "tidied": False,
            "terms": len(glossary),
            "names_fixed": fixed,
        }
    ]
    used += extra
    texts, stats = {}, new_repair_stats()
    progress("translate", 0, len(sections))
    for done, item in enumerate(sections, start=1):
        text, tokens, truncated, attempts = translate_section(item["text"], label, lang, model, glossary, titles, stats)
        text, tidied = tidy_section(localize_labels(text.strip(), names, titles))
        used += tokens
        found = language.audit(text)
        report.append(
            {
                "section": f"translate:{item['key']}",
                "source": None,
                "tokens": tokens,
                "truncated": truncated,
                "tidied": tidied,
                "attempts": attempts,
                "english_share": found["english_share"],
                "english_left": found["english_paragraphs"] + found["english_tables"] + found["english_lines"],
                "english_labels": len(english_labels(text, titles)),
            }
        )
        texts[item["key"]] = text
        progress("translate", done, len(sections))
    report.append(
        {"section": "translate:repairs", "source": None, "tokens": 0, "truncated": False, "tidied": False, **stats}
    )
    return texts, report, used, glossary


def _call(prompt: str, model: str, max_tokens: int) -> tuple[str, int, bool]:
    budget = max(400, min(max_tokens, TOKEN_BUDGET - len(prompt) // 3))
    text, tokens, finish = llm.complete(prompt, model=model, max_tokens=budget)
    if finish == "length" and text.strip():
        more_prompt = prompt + CONTINUE.format(tail=text[-1200:])
        budget = max(400, min(max_tokens, TOKEN_BUDGET - len(more_prompt) // 3))
        more, extra, finish = llm.complete(more_prompt, model=model, max_tokens=budget)
        text, tokens = text.rstrip() + " " + more.lstrip(), tokens + extra
    return text, tokens, finish == "length"


DOCUMENT_RULES = (
    "\n- Never mention summaries, cards, material, context sources or how this text was produced."
    "\n- A sentence that names a work must carry that work's own number. Do not claim that a work does "
    "not address something unless the material says so."
    "\n- The research topic is exactly: {topic}. Writing in a given language does not make the research "
    "about that language. Never narrow the topic to a language, country or domain it does not name; if "
    "you need a novel angle, derive it from the gaps the material states."
    "\n- Derive gaps and limitations only from what the material states."
)


def section(key, en, tr, kind, task, tokens, cited=True, last=False):
    return {
        "key": key,
        "title": {"en": en, "tr": tr},
        "kind": kind,
        "task": task,
        "tokens": tokens,
        "cited": cited,
        "last": last,
    }


THEMES = "THEMES"
DOCUMENT_TYPES = {
    "survey": {
        "label": "literature review",
        "theme_prefix": {"en": "", "tr": ""},
        "theme_task": "Write this thematic section as a connected narrative: the problem the works share, how their "
        "approaches differ and the results they report. Give the works with detailed material more "
        "depth, and mention every listed work at least once with its own citation.",
        "sections": [
            section(
                "intro",
                "Introduction",
                "Giriş",
                "overview",
                "Motivate the topic, state the scope (this review covers {n} works organized into {k} themes and "
                "does not cover the whole field) and outline the themes in their order.",
                1500,
            ),
            section(
                "background",
                "Background",
                "Arka Plan",
                "brief",
                "Explain the concepts and terminology a reader needs before the themes, citing the works they "
                "come from.",
                1500,
            ),
            THEMES,
            section(
                "comparison",
                "Comparison and Discussion",
                "Karşılaştırma ve Tartışma",
                "setup",
                "Compare the representative works across the themes. Include one compact markdown table with the "
                "columns Work, Year, Key idea, Setting or scale and Main result: one row for every theme and at most "
                "8 rows, every row citing its work, and every cell under 12 words. After the table, write two short "
                "paragraphs that discuss the trade-offs between the approaches.",
                3500,
            ),
            section(
                "open",
                "Open Problems",
                "Açık Problemler",
                "limits",
                "Derive open problems and research directions from the limitations and gaps of the works. Tie "
                "every problem to the works that motivate it, with [n].",
                1500,
            ),
            section(
                "conclusion",
                "Conclusion",
                "Sonuç",
                "written",
                "Summarize the main insights of this review without introducing new claims.",
                900,
                cited=False,
            ),
        ],
    },
    "proposal": {
        "label": "research proposal",
        "theme_prefix": {"en": "Related Work: ", "tr": "İlgili Çalışmalar: "},
        "theme_task": "Review the works of this theme as related work for the proposal: what they achieve, how they "
        "differ and which limitation or gap the material states for them. Mention every listed work "
        "at least once with its own citation.",
        "sections": [
            section(
                "abstract",
                "Abstract",
                "Özet",
                "written",
                "Write the abstract of the proposal: the problem, the gap, the proposed approach, the evaluation "
                "and the expected contributions, in one or two paragraphs.",
                800,
                cited=False,
                last=True,
            ),
            section(
                "problem",
                "Problem and Motivation",
                "Problem ve Motivasyon",
                "overview",
                "State the problem and why it matters. Use the themes and works to show what is already solved "
                "and what remains open, citing the works behind every statement. Mention that the proposal builds "
                "on {n} works organized into {k} themes.",
                1500,
            ),
            THEMES,
            section(
                "questions",
                "Research Questions",
                "Araştırma Soruları",
                "limits",
                "Formulate three to five research questions that follow from the gaps identified in the related "
                "work. Explain each one and cite the works whose limitations motivate it.",
                1200,
            ),
            section(
                "method",
                "Proposed Method",
                "Önerilen Yöntem",
                "full",
                "Describe the proposed new method as a plan: its components, data, training and how it answers "
                "each research question. Cite works whose components you build on. Never present results.",
                2500,
                cited=False,
            ),
            section(
                "evaluation",
                "Evaluation Plan",
                "Değerlendirme Planı",
                "setup",
                "Describe the evaluation plan: data sets, baselines taken from the cited works, metrics and ablations.",
                1800,
                cited=False,
            ),
            section(
                "contributions",
                "Expected Contributions",
                "Beklenen Katkılar",
                "written",
                "State the expected contributions of the proposed work.",
                1000,
                cited=False,
            ),
            section(
                "risks",
                "Risks and Limitations",
                "Riskler ve Sınırlar",
                "limits",
                "Discuss the risks and limitations of the plan and how they will be mitigated.",
                1200,
                cited=False,
            ),
        ],
    },
    "synthesis": {
        "label": "synthesis report",
        "theme_prefix": {"en": "Findings: ", "tr": "Bulgular: "},
        "theme_task": "Synthesize the findings of the works in this theme: what they show together, with their "
        "numbers and conditions, where results agree or conflict, and the evidence behind them. "
        "Mention every listed work at least once with its own citation.",
        "sections": [
            section(
                "summary",
                "Summary",
                "Özet",
                "written",
                "Summarize what this body of work shows when taken together, in one or two paragraphs.",
                800,
                cited=False,
                last=True,
            ),
            THEMES,
            section(
                "agreements",
                "Agreements and Differences",
                "Ortak Noktalar ve Farklar",
                "setup",
                "Compare the findings across the themes point by point: where they agree, where they conflict and "
                "why. Include one markdown table with the columns Work, Year, Setting and Main finding, with at "
                "least one row for every theme, where every row cites its work.",
                2500,
            ),
            section(
                "implications",
                "Implications",
                "Çıkarımlar",
                "limits",
                "Discuss the implications for research and practice. Tie every implication to the findings it "
                "builds on, with [n].",
                1500,
            ),
            section(
                "conclusion",
                "Conclusion",
                "Sonuç",
                "written",
                "Conclude the report without introducing new claims.",
                800,
                cited=False,
            ),
        ],
    },
}


def uncited_allowed(doc_type: str, lang: str) -> list[str]:
    return [sec["title"][lang] for sec in DOCUMENT_TYPES[doc_type]["sections"] if sec != THEMES and not sec["cited"]]


def work_line(s: dict) -> str:
    return card_text(s) if s.get("card") else f"[{s['n']}] {s['title']} ({_year(s)}): {s.get('key_sentences', '')}"


def material_for(kind: str, sources: list[dict], themes: list[dict], written: list, theme=None) -> str:
    deep = [s for s in sources if s.get("card")]
    so_far = "\n\nWritten so far:\n" + summary_of(written)
    if kind == "theme":
        return "\n\n".join(work_line(s) for s in sources if s["id"] in theme["ids"])
    if kind == "overview":
        lines = [f"Themes ({len(themes)}):"] + [f"- {t['name']}: {t['description']}" for t in themes]
        return "\n".join(lines) + "\n\n" + "\n\n".join(card_text(s, ("problem", "method")) for s in deep)
    if kind == "brief":
        return "\n\n".join(card_text(s, ("problem", "method")) for s in deep)
    if kind == "full":
        return "\n\n".join(card_text(s, limit=350) for s in deep) + so_far
    if kind == "setup":
        others = [s for t in themes for s in [x for x in sources if x["id"] in t["ids"] and not x.get("card")][:3]]
        lines = "\n".join(work_line(s) for s in others)
        return (
            "\n\n".join(card_text(s, ("method", "setup", "findings"), limit=500) for s in deep)
            + ("\n\nOther representative works by theme:\n" + lines if lines else "")
            + so_far
        )
    if kind == "limits":
        return "\n\n".join(card_text(s, ("findings", "limitations")) for s in deep) + so_far
    return so_far.strip()


def clean_title(title: str, fallback: str) -> str:
    title = re.sub(r"[\u3010\u3011\u300c\u300d\u300e\u300f\u300a\u300b]", "", title)
    return title.strip().strip('"').strip("#").strip() or fallback


def local_names(themes: list[dict], lang: str, model: str) -> tuple[dict, int, int]:
    english = [t["name"] for t in themes]
    if lang == "en":
        return dict(zip(english, english)), 0, 0
    local = {t["name"]: (t.get("name_local") or "").strip() for t in themes}
    missing = [en for en in english if not local[en] or same_text(local[en], en) or language.english_words(local[en])]
    translated, tokens = translate_lines(missing, lang, model)
    local.update(zip(missing, translated))
    return local, tokens, len(english) - len(missing)


def write_document(
    doc_type: str, sources: list[dict], themes: list[dict], topic: str, lang: str, model: str, progress=None
) -> tuple[str, int, list[dict], dict]:
    progress = progress or (lambda stage, done, total: None)
    spec, n = DOCUMENT_TYPES[doc_type], len(sources)
    label = f"{spec['label']} on {topic}"
    plan = []
    local, heading_tokens, from_plan = local_names(themes, lang, model)
    for sec in spec["sections"]:
        if sec == THEMES:
            plan += [
                {
                    "key": f"theme:{t['name']}",
                    "kind": "theme",
                    "task": spec["theme_task"],
                    "tokens": 2500,
                    "last": False,
                    "theme": t,
                    "heading": spec["theme_prefix"]["en"] + t["name"],
                    "heading_out": spec["theme_prefix"][lang] + local[t["name"]],
                }
                for t in themes
            ]
        else:
            plan.append({**sec, "theme": None, "heading": sec["title"]["en"], "heading_out": sec["title"][lang]})
    outline = "\n".join(f"## {p['heading']}" for p in plan)
    texts, written, report, used = {}, [], [], heading_tokens
    if lang != "en" and themes:
        report.append(
            {
                "section": "translate:headings",
                "source": None,
                "tokens": heading_tokens,
                "truncated": False,
                "tidied": False,
                "translated": sum(not same_text(local[t["name"]], t["name"]) for t in themes),
                "of": len(themes),
                "from_plan": from_plan,
            }
        )
    progress("sections", 0, len(plan))
    for item in [p for p in plan if not p["last"]] + [p for p in plan if p["last"]]:
        prompt = SECTION_PROMPT.format(
            label=label,
            language="English",
            outline=outline,
            written=summary_of(written),
            target=f'the section "{item["heading"]}"',
            task=item["task"].format(n=n, k=len(themes)),
            material=material_for(item["kind"], sources, themes, written, item["theme"]),
            style=STYLE,
            n=n,
            extra=DOCUMENT_RULES.format(topic=topic),
        )
        text, tokens, truncated = _call(prompt, model, item["tokens"])
        text, tidied = tidy_section(text.strip(), truncated)
        text, repeated = drop_repeated_heading(
            text, [item["heading"], item["theme"]["name"] if item["theme"] else None]
        )
        if lang != "en":
            text = unbold_labels(text, [s["title"] for s in sources])
        used += tokens
        report.append(
            {
                "section": item["key"],
                "source": None,
                "tokens": tokens,
                "truncated": truncated,
                "tidied": tidied or repeated,
                "trimmed": truncated and tidied,
            }
        )
        texts[item["key"]] = text
        written.append((item["heading"], text))
        progress("sections", len(written), len(plan))
    draft = {
        "label": label,
        "lang": lang,
        "names": local,
        "titles": [s["title"] for s in sources],
        "glossary": {},
        "sections": [
            {"key": p["key"], "heading": p["heading"], "heading_out": p["heading_out"], "text": texts[p["key"]]}
            for p in plan
        ],
    }
    if lang != "en":
        texts, rows, tokens, draft["glossary"] = translate_sections(
            draft["sections"], label, lang, model, local, progress, draft["titles"]
        )
        report += rows
        used += tokens
        for p, section in zip(plan, draft["sections"]):
            if p["theme"]:
                p["heading_out"] = section["heading_out"] = spec["theme_prefix"][lang] + local[p["theme"]["name"]]
    title, tokens, _ = _call(
        TITLE_PROMPT.format(language=LANGUAGES[lang], label=label, written=summary_of(written)), model, 300
    )
    used += tokens
    title = clean_title(title, topic)
    draft["title"] = title
    body = "\n\n".join(f"## {p['heading_out']}\n\n{texts[p['key']]}" for p in plan)
    return f"# {title}\n\n{body}\n", used, report, draft
