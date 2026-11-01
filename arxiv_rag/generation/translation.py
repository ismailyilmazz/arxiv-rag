import re

from arxiv_rag.generation import gemini, language, llm
from arxiv_rag.generation.formatting import _citation_numbers, strip_meta_notes, tidy_section
from arxiv_rag.generation.language import LANGUAGES
from arxiv_rag.generation.llm import complete_continued

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
    text, tokens, _ = complete_continued(prompt, model, REPAIR_TOKENS)
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
        or glossary_left(block, glossary, titles)
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
            if _accepted(piece, block, titles) and not glossary_left(piece, glossary, titles):
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
    out, tokens, truncated = complete_continued(
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
    if language.has_english(out) or english_labels(out, titles) or glossary_left(out, glossary, titles):
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
    return max(words, key=len)[:5] if words else ""


def _flat(text: str) -> str:
    return " ".join(language.normalize(text).replace("-", " ").split())


def _term_pattern(term: str) -> str:
    stem = term[:-1] if term.endswith("s") and len(term) > 4 else term
    return rf"(?<![a-z]){re.escape(stem)}s?(?![a-z])"


def glossary_left(text: str, glossary, titles=()) -> bool:
    if not glossary:
        return False
    plain = " " + " ".join(language._translatable(text).lower().replace("-", " ").split()) + " "
    for title in titles:
        plain = plain.replace(_flat(title), " ")
    for key, target in glossary.items():
        term = _flat(key)
        if not term or term == _flat(target) or key.isupper() or len(term) < 4:
            continue
        if re.search(_term_pattern(term), plain):
            return True
    return False


def misaligned(english: str, local: str, glossary: dict, skip=()) -> bool:
    name, lowered = _flat(english), local.lower()
    for key, target in glossary.items():
        term = _flat(key)
        if key in skip or term == name or len(term) < 4 or same_text(key, target):
            continue
        stem = _stem(target)
        if re.search(_term_pattern(term), name) and len(stem) >= 4 and stem not in lowered:
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
