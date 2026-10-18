import re

from arxiv_rag.generation import language, llm

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
- Keep the markdown formatting (tables, bold text, lists) unchanged.
- Keep paper titles, model names and standard technical terms in English.
- Write natural, fluent {language}; in Turkish every sentence ends with its predicate.
- Do not add, remove or summarize content. Return only the translated section.

Section:
{text}"""

RETRY_TRANSLATE_PROMPT = """The previous translation attempt returned English. Translate the text below into {language}.
- Every sentence of your output must be in {language}. Only paper titles, model names and standard technical terms
  (for example state-of-the-art, AUC, embedding, token) may stay in English.
- Keep every citation such as [3] or [1, 4] exactly as it is and where it is, and keep the markdown formatting.
- Do not add notes or comments about the text. Return only the translation.

Text:
{text}"""

HEADINGS_PROMPT = """Translate each line below into {language}. Keep model names and standard technical terms in English.
Return exactly {n} lines in the same order and nothing else.

{lines}"""

MAX_ENGLISH_SHARE = 0.0
META_NOTE = re.compile(
    r"\s*\((?=[^()]*(?:çevir|translat))(?=[^()]*(?:eksik|tamamlan|yapılamad|incomplete|missing|could not))[^()]{5,300}\)"
)

_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_TERMINAL = re.compile(r"[.!?\])\"'\u201d]\s*$")


def _cells(line: str) -> tuple:
    return tuple(c.strip().lower() for c in line[line.index("|") :].strip().strip("|").split("|"))


def tidy_section(text: str) -> tuple[str, bool]:
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
    body = "\n".join(lines).rstrip()
    last = body.splitlines()[-1].strip() if body else ""
    if (
        last
        and not last.startswith(("|", "-", "*", "#"))
        and not re.match(r"\d+\.", last)
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


def is_complete(candidate: str, source: str, min_ratio: float = 0.7) -> bool:
    return _citation_numbers(source) <= _citation_numbers(candidate) and len(candidate.strip()) >= min_ratio * len(
        source.strip()
    )


def repair_paragraphs(text: str, lang: str, model: str, tables: bool = False) -> tuple[str, int, int, int]:
    blocks, tokens, repaired, failed = re.split(r"\n\s*\n", text.strip()), 0, 0, 0
    for i, block in enumerate(blocks):
        is_table = block.lstrip().startswith("|")
        prose = " ".join(line for line in block.splitlines() if not line.lstrip().startswith(("|", "#")))
        wanted = (tables and is_table) or (
            not is_table and len(prose.split()) >= 8 and language.paragraph_is_english(prose)
        )
        if not wanted:
            continue
        prompt = RETRY_TRANSLATE_PROMPT.format(language=LANGUAGES[lang], text=block)
        for _ in range(2):
            piece, extra, _ = _call(prompt, model, 2000)
            piece, tokens = strip_meta_notes(piece.strip()), tokens + extra
            if piece and is_complete(piece, block) and (is_table or not language.paragraph_is_english(piece)):
                blocks[i], repaired = piece, repaired + 1
                break
        else:
            failed += 1
    return "\n\n".join(blocks), tokens, repaired, failed


def translate_section(text: str, label: str, lang: str, model: str) -> tuple[str, int, bool, int]:
    out, tokens, truncated = _call(
        TRANSLATE_PROMPT.format(label=label, language=LANGUAGES[lang], text=text), model, 4000
    )
    out = strip_meta_notes(out)
    if lang == "en":
        return out, tokens, truncated, 1
    if truncated or not is_complete(out, text):
        out, extra, _, _ = repair_paragraphs(text, lang, model, tables=True)
        return out, tokens + extra, False, 2
    if language.english_share(out) > MAX_ENGLISH_SHARE:
        out, extra, repaired, _ = repair_paragraphs(out, lang, model)
        return out, tokens + extra, truncated, 2 if repaired else 1
    return out, tokens, truncated, 1


def translate_lines(lines: list[str], lang: str, model: str) -> tuple[list[str], int]:
    if lang == "en" or not lines:
        return list(lines), 0
    prompt = HEADINGS_PROMPT.format(language=LANGUAGES[lang], n=len(lines), lines="\n".join(lines))
    out, tokens, _ = _call(prompt, model, 400)
    got = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip('"') for line in out.splitlines() if line.strip()]
    return (got if len(got) == len(lines) and all(got) else list(lines)), tokens


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
                "Compare the representative works across the themes. Include one markdown table with the columns "
                "Work, Year, Key idea, Setting or scale and Main result, with at least one row for every theme, "
                "where every row cites its work. Then discuss the trade-offs.",
                2500,
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


def write_document(
    doc_type: str, sources: list[dict], themes: list[dict], topic: str, lang: str, model: str, progress=None
) -> tuple[str, int, list[dict]]:
    progress = progress or (lambda stage, done, total: None)
    spec, n = DOCUMENT_TYPES[doc_type], len(sources)
    label = f"{spec['label']} on {topic}"
    plan = []
    names, heading_tokens = translate_lines([t["name"] for t in themes], lang, model)
    local = dict(zip([t["name"] for t in themes], names))
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
        text, tidied = tidy_section(text.strip())
        used += tokens
        report.append(
            {"section": item["key"], "source": None, "tokens": tokens, "truncated": truncated, "tidied": tidied}
        )
        texts[item["key"]] = text
        written.append((item["heading"], text))
        progress("sections", len(written), len(plan))
    if lang != "en":
        progress("translate", 0, len(plan))
        for done, item in enumerate(plan, start=1):
            text, tokens, truncated, attempts = translate_section(texts[item["key"]], label, lang, model)
            text, tidied = tidy_section(text.strip())
            used += tokens
            report.append(
                {
                    "section": f"translate:{item['key']}",
                    "source": None,
                    "tokens": tokens,
                    "truncated": truncated,
                    "tidied": tidied,
                    "attempts": attempts,
                    "english_share": round(language.english_share(text), 3),
                }
            )
            texts[item["key"]] = text
            progress("translate", done, len(plan))
    title, tokens, _ = _call(
        TITLE_PROMPT.format(language=LANGUAGES[lang], label=label, written=summary_of(written)), model, 300
    )
    used += tokens
    title = title.strip().strip('"').strip("#").strip() or topic
    body = "\n\n".join(f"## {p['heading_out']}\n\n{texts[p['key']]}" for p in plan)
    return f"# {title}\n\n{body}\n", used, report
