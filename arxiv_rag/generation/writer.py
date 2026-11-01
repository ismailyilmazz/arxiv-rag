import re

from arxiv_rag.generation import gemini, language
from arxiv_rag.generation.formatting import (
    _citation_numbers,
    drop_repeated_heading,
    normalize_citations,
    strip_meta_notes,
    tidy_section,
)
from arxiv_rag.generation.language import LANGUAGES
from arxiv_rag.generation.llm import complete_continued
from arxiv_rag.generation.translation import (
    GEMINI_REPAIR_TOKENS,
    _repair_engines,
    same_text,
    translate_lines,
    translate_sections,
    unbold_labels,
)

STYLE = (
    "Write complete sentences, each with an explicit subject and a finite verb; in Turkish every sentence ends "
    "with its predicate. Prefer flowing paragraphs; use bullet points only for short enumerations and write "
    "each bullet as a full sentence."
)


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


COMPLETE_PROMPT = """{prompt}

The section below was cut off before it was finished. Continue it from exactly where it stops, starting with the next
sentence (or the next table row if it stops inside a table). Write only the missing remainder: do not repeat, rewrite
or summarize what is already written, and do not add a heading. Follow every rule of the task above, including the
citation format [n] with the numbers of the given sources only.

Section so far:
{text}"""


CONTINUATION_MIN_WORDS = 15


CONTINUATION_MAX_WORDS = 1500


def _shingles(text: str, k: int = 8) -> set:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def accept_continuation(extra: str, base: str, n_sources: int) -> bool:
    words = len(extra.split())
    if not CONTINUATION_MIN_WORDS <= words <= max(CONTINUATION_MAX_WORDS, len(base.split())):
        return False
    if extra.lstrip().startswith("#") or not _citation_numbers(extra) <= set(range(1, n_sources + 1)):
        return False
    new = _shingles(extra)
    return not new or len(new & _shingles(base)) / len(new) <= 0.3


def complete_section(prompt: str, text: str, n_sources: int) -> tuple[str, bool, bool]:
    base, _ = tidy_section(text.strip(), True)
    if not base or "gemini" not in _repair_engines():
        return text, False, True
    try:
        extra, _, finish = gemini.complete(
            COMPLETE_PROMPT.format(prompt=prompt, text=base), max_tokens=GEMINI_REPAIR_TOKENS
        )
    except gemini.GeminiUnavailable:
        return text, False, True
    extra = normalize_citations(strip_meta_notes(extra.strip()))
    if not accept_continuation(extra, base, n_sources):
        return text, False, True
    last = base.splitlines()[-1].lstrip()
    joiner = (
        "\n" if extra.startswith("|") and last.startswith("|") else "\n\n" if extra.startswith(("|", "-", "*")) else " "
    )
    return base + joiner + extra, True, finish == "length"


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
        text, tokens, truncated = complete_continued(prompt, model, item["tokens"])
        completed = False
        if truncated:
            text, completed, truncated = complete_section(prompt, text, n)
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
                "completed": completed,
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
    title, tokens, _ = complete_continued(
        TITLE_PROMPT.format(language=LANGUAGES[lang], label=label, written=summary_of(written)), model, 300
    )
    used += tokens
    title = clean_title(title, topic)
    draft["title"] = title
    body = "\n\n".join(f"## {p['heading_out']}\n\n{texts[p['key']]}" for p in plan)
    return f"# {title}\n\n{body}\n", used, report, draft
