import re

from core import llm

LANGUAGES = {"en": "English", "tr": "Turkish"}
STYLE = ("Write complete sentences, each with an explicit subject and a finite verb; in Turkish every sentence ends "
         "with its predicate. Prefer flowing paragraphs; use bullet points only for short enumerations and write "
         "each bullet as a full sentence.")
TOKEN_BUDGET = 7600


def section(key, en, tr, material, task, max_tokens, last=False, cited=True):
    return {"key": key, "title": {"en": en, "tr": tr}, "material": material, "task": task,
            "max_tokens": max_tokens, "last": last, "cited": cited}


DOC_TYPES = {
    "survey": {"label": "focused literature review", "sections": [
        section("intro", "Introduction", "Giriş", ["brief", "context"],
                "Motivate the topic, state clearly that this review covers the selected sources rather than the "
                "whole field, and outline the following sections.", 1500),
        section("background", "Background", "Arka Plan", ["brief", "context"],
                "Explain the concepts, terminology and earlier ideas a reader needs before the individual "
                "approaches, and cite the source each concept comes from.", 1500),
        section("approaches", "Approaches", "Yaklaşımlar", ["source"],
                "Describe this source in depth: the problem it addresses, how its method works including the key "
                "design choices, its experimental set-up, its results with their exact numbers and its stated "
                "limitations. Use the excerpt for details the card does not contain.", 2500),
        section("comparison", "Comparison and Discussion", "Karşılaştırma ve Tartışma", ["full", "context"],
                "Compare the sources along design, training and data, efficiency, results and limitations. Use a "
                "level-3 heading for each dimension and point out agreements, disagreements and trade-offs.", 2500),
        section("open", "Open Problems", "Açık Problemler", ["limits", "context"],
                "Derive open problems and research directions from the limitations and gaps of the sources. Tie "
                "every problem to the source or sources whose limitation motivates it, with [n].", 1500),
        section("conclusion", "Conclusion", "Sonuç", ["written"],
                "Summarize the main insights of this survey without introducing new claims.", 1000, cited=False),
    ]},
    "proposal": {"label": "research proposal", "sections": [
        section("abstract", "Abstract", "Özet", ["written"],
                "Write the abstract of the proposal: the problem, the gap, the proposed approach, the evaluation and "
                "the expected contributions, in one or two paragraphs.", 800, last=True, cited=False),
        section("problem", "Problem and Motivation", "Problem ve Motivasyon", ["brief", "limits", "context"],
                "State the problem and why it matters; use the sources to show what is already solved and what "
                "remains open, citing the source behind every statement.", 1500),
        section("related", "Related Work", "İlgili Çalışmalar", ["source"],
                "Discuss this source as related work: its method, set-up and results with their numbers, and the "
                "limitation that matters for this proposal.", 2000),
        section("questions", "Research Questions", "Araştırma Soruları", ["limits", "written", "context"],
                "Formulate three to five research questions that follow from the gaps, and explain each one.", 1200),
        section("method", "Proposed Method", "Önerilen Yöntem", ["full", "written"],
                "Describe the proposed new method as a plan: its components, data, training and how it answers each "
                "research question. Never present results.", 2500, cited=False),
        section("evaluation", "Evaluation Plan", "Değerlendirme Planı", ["full", "written"],
                "Describe the evaluation plan: data sets, baselines taken from the sources, metrics and ablations.",
                1800, cited=False),
        section("contributions", "Expected Contributions", "Beklenen Katkılar", ["written"],
                "State the expected contributions of the proposed work.", 1000, cited=False),
        section("risks", "Risks and Limitations", "Riskler ve Sınırlar", ["written", "limits"],
                "Discuss the risks and limitations of the plan and how they will be mitigated.", 1200, cited=False),
    ]},
    "synthesis": {"label": "synthesis report", "sections": [
        section("summary", "Summary", "Özet", ["written"],
                "Summarize what these sources show when taken together, in one or two paragraphs.", 800, last=True,
                cited=False),
        section("findings", "Key Findings", "Temel Bulgular", ["source"],
                "Present the findings of this source with their numbers and conditions, and the evidence behind "
                "them.", 2000),
        section("agreements", "Agreements and Differences", "Ortak Noktalar ve Farklar", ["full", "context"],
                "Compare the sources point by point: where they agree, where they differ and why.", 2000),
        section("implications", "Implications", "Çıkarımlar", ["full", "written", "context"],
                "Discuss the implications of these findings for research and practice. Tie every implication to "
                "the findings it builds on, with [n].", 1500),
        section("conclusion", "Conclusion", "Sonuç", ["written"],
                "Conclude the report without introducing new claims.", 800, cited=False),
    ]},
}

SECTION_PROMPT = """You are writing one section of a {label} in {language}, based only on numbered sources.

Document outline:
{outline}

Already written:
{written}

Now write: {target}
Task: {task}{context_rule}

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


def card_text(s: dict, keys=("problem", "method", "setup", "findings", "limitations")) -> str:
    lines = [f"[{s['n']}] {s['title']} ({_year(s)})"]
    lines += [f"{k.capitalize()}: {s['card'][k]}" for k in keys if s["card"].get(k)]
    return "\n".join(lines)


def summary_of(written: list[tuple[str, str]]) -> str:
    if not written:
        return "Nothing yet."
    lines = []
    for heading, text in written:
        firsts = [re.split(r"(?<=[.!?])\s", p.strip(), maxsplit=1)[0]
                  for p in re.split(r"\n\s*\n", text) if p.strip() and not p.lstrip().startswith("#")]
        lines.append(f"{heading}: " + " ".join(firsts)[:600])
    return "\n".join(lines)


def brief_abstract(text: str, sentences: int = 2, max_chars: int = 320) -> str:
    parts = re.split(r"(?<=[.!?])\s+", " ".join((text or "").split()))
    return " ".join(parts[:sentences])[:max_chars]


def context_text(context: list[dict]) -> str:
    lines = ["Context sources (known only from these short summaries):"]
    lines += [f"[{c['n']}] {c['title']} ({_year(c)}): {brief_abstract(c.get('abstract'))}" for c in context]
    return "\n".join(lines)


def material_for(spec: dict, sources: list[dict], written: list[tuple[str, str]], source=None,
                 context: list[dict] = ()) -> str:
    if source is not None:
        parts = [card_text(source)]
        if source.get("excerpt"):
            parts.append(f"Excerpt from [{source['n']}]:\n{source['excerpt']}")
        return "\n\n".join(parts)
    parts = []
    for kind in spec["material"]:
        if kind == "brief":
            parts += [card_text(s, ("problem", "method")) for s in sources]
        elif kind == "full":
            parts += [card_text(s) for s in sources]
        elif kind == "limits":
            parts += [card_text(s, ("findings", "limitations")) for s in sources]
        elif kind == "written":
            parts.append("Written so far:\n" + summary_of(written))
        elif kind == "context" and context:
            parts.append(context_text(context))
    return "\n\n".join(parts)


def _call(prompt: str, model: str, max_tokens: int) -> tuple[str, int, bool]:
    budget = max(400, min(max_tokens, TOKEN_BUDGET - len(prompt) // 3))
    text, tokens, finish = llm.complete(prompt, model=model, max_tokens=budget)
    return text, tokens, finish == "length"


def write_document(sources: list[dict], doc_type: str, lang: str, model: str,
                   context: list[dict] = ()) -> tuple[str, int, list[dict]]:
    spec, language = DOC_TYPES[doc_type], LANGUAGES[lang]
    n_total = len(sources) + len(context)
    outline = "\n".join(f"## {s['title'][lang]}" for s in spec["sections"])
    order = [s for s in spec["sections"] if not s["last"]] + [s for s in spec["sections"] if s["last"]]
    written, parts, report, total = [], {}, [], 0

    for sec in order:
        heading = sec["title"][lang]
        targets = [(f"the subsection about source [{s['n']}] in the section \"{heading}\"", s)
                   for s in sources] if sec["material"] == ["source"] else [(f"the section \"{heading}\"", None)]
        chunks = []
        for target, source in targets:
            with_context = bool(context) and "context" in sec["material"] and source is None
            rule = (f"\nSources 1 to {len(sources)} are reviewed in depth. Sources {len(sources) + 1} to {n_total} "
                    f"are context sources known only from short summaries: cite them only for what their summary "
                    f"states." if with_context else "")
            prompt = SECTION_PROMPT.format(
                label=spec["label"], language=language, outline=outline, written=summary_of(written),
                target=target, task=sec["task"], context_rule=rule,
                material=material_for(sec, sources, written, source, context if with_context else ()),
                style=STYLE, n=n_total, extra="")
            text, tokens, truncated = _call(prompt, model, sec["max_tokens"])
            total += tokens
            report.append({"section": sec["key"], "source": source["n"] if source else None,
                           "tokens": tokens, "truncated": truncated})
            if source is not None:
                text = f"### [{source['n']}] {source['title']} ({_year(source)})\n\n{text}"
            chunks.append(text.strip())
        parts[sec["key"]] = "\n\n".join(chunks)
        written.append((heading, parts[sec["key"]]))

    title, tokens, _ = _call(TITLE_PROMPT.format(language=language, label=spec["label"],
                                                 written=summary_of(written)), model, 300)
    total += tokens
    title = title.strip().strip('"').strip("#").strip() or spec["label"].title()
    body = "\n\n".join(f"## {s['title'][lang]}\n\n{parts[s['key']]}" for s in spec["sections"])
    return f"# {title}\n\n{body}\n", total, report


def uncited_allowed(doc_type: str, lang: str) -> list[str]:
    return [sec["title"][lang] for sec in DOC_TYPES[doc_type]["sections"] if not sec["cited"]]


BROAD_RULES = ("\n- Never mention summaries, cards, material, context sources or how this text was produced."
               "\n- A sentence that names a work must carry that work's own number. Do not claim that a work does "
               "not address something unless the material says so.")


def fixed(key, en, tr, kind, task, tokens, cited=True, last=False):
    return {"key": key, "title": {"en": en, "tr": tr}, "kind": kind, "task": task, "tokens": tokens,
            "cited": cited, "last": last}


THEMES = "THEMES"
BROAD = {
    "survey": {
        "label": "literature review", "theme_prefix": {"en": "", "tr": ""},
        "theme_task": "Write this thematic section as a connected narrative: the problem the works share, how their "
                      "approaches differ and the results they report. Give the works with detailed material more "
                      "depth, and mention every listed work at least once with its own citation.",
        "sections": [
            fixed("intro", "Introduction", "Giriş", "overview",
                  "Motivate the topic, state the scope (this review covers {n} works organized into {k} themes and "
                  "does not cover the whole field) and outline the themes in their order.", 1500),
            fixed("background", "Background", "Arka Plan", "brief",
                  "Explain the concepts and terminology a reader needs before the themes, citing the works they "
                  "come from.", 1500),
            THEMES,
            fixed("comparison", "Comparison and Discussion", "Karşılaştırma ve Tartışma", "setup",
                  "Compare the representative works across the themes. Include one markdown table with the columns "
                  "Work, Year, Key idea, Setting or scale and Main result, where every row cites its work. Then "
                  "discuss the trade-offs.", 2500),
            fixed("open", "Open Problems", "Açık Problemler", "limits",
                  "Derive open problems and research directions from the limitations and gaps of the works. Tie "
                  "every problem to the works that motivate it, with [n].", 1500),
            fixed("conclusion", "Conclusion", "Sonuç", "written",
                  "Summarize the main insights of this review without introducing new claims.", 900, cited=False),
        ]},
    "proposal": {
        "label": "research proposal", "theme_prefix": {"en": "Related Work: ", "tr": "İlgili Çalışmalar: "},
        "theme_task": "Review the works of this theme as related work for the proposal: what they achieve, how they "
                      "differ and which limitation or gap they leave open that matters for this topic. Mention "
                      "every listed work at least once with its own citation.",
        "sections": [
            fixed("abstract", "Abstract", "Özet", "written",
                  "Write the abstract of the proposal: the problem, the gap, the proposed approach, the evaluation "
                  "and the expected contributions, in one or two paragraphs.", 800, cited=False, last=True),
            fixed("problem", "Problem and Motivation", "Problem ve Motivasyon", "overview",
                  "State the problem and why it matters. Use the themes and works to show what is already solved "
                  "and what remains open, citing the works behind every statement. Mention that the proposal builds "
                  "on {n} works organized into {k} themes.", 1500),
            THEMES,
            fixed("questions", "Research Questions", "Araştırma Soruları", "limits",
                  "Formulate three to five research questions that follow from the gaps identified in the related "
                  "work. Explain each one and cite the works whose limitations motivate it.", 1200),
            fixed("method", "Proposed Method", "Önerilen Yöntem", "full",
                  "Describe the proposed new method as a plan: its components, data, training and how it answers "
                  "each research question. Cite works whose components you build on. Never present results.", 2500,
                  cited=False),
            fixed("evaluation", "Evaluation Plan", "Değerlendirme Planı", "setup",
                  "Describe the evaluation plan: data sets, baselines taken from the cited works, metrics and "
                  "ablations.", 1800, cited=False),
            fixed("contributions", "Expected Contributions", "Beklenen Katkılar", "written",
                  "State the expected contributions of the proposed work.", 1000, cited=False),
            fixed("risks", "Risks and Limitations", "Riskler ve Sınırlar", "limits",
                  "Discuss the risks and limitations of the plan and how they will be mitigated.", 1200,
                  cited=False),
        ]},
    "synthesis": {
        "label": "synthesis report", "theme_prefix": {"en": "Findings: ", "tr": "Bulgular: "},
        "theme_task": "Synthesize the findings of the works in this theme: what they show together, with their "
                      "numbers and conditions, where results agree or conflict, and the evidence behind them. "
                      "Mention every listed work at least once with its own citation.",
        "sections": [
            fixed("summary", "Summary", "Özet", "written",
                  "Summarize what this body of work shows when taken together, in one or two paragraphs.", 800,
                  cited=False, last=True),
            THEMES,
            fixed("agreements", "Agreements and Differences", "Ortak Noktalar ve Farklar", "setup",
                  "Compare the findings across the themes point by point: where they agree, where they conflict and "
                  "why. Include one markdown table with the columns Work, Year, Setting and Main finding, where "
                  "every row cites its work.", 2500),
            fixed("implications", "Implications", "Çıkarımlar", "limits",
                  "Discuss the implications for research and practice. Tie every implication to the findings it "
                  "builds on, with [n].", 1500),
            fixed("conclusion", "Conclusion", "Sonuç", "written",
                  "Conclude the report without introducing new claims.", 800, cited=False),
        ]},
}


def broad_uncited_allowed(doc_type: str, lang: str) -> list[str]:
    return [sec["title"][lang] for sec in BROAD[doc_type]["sections"] if sec != THEMES and not sec["cited"]]


def work_line(s: dict) -> str:
    return card_text(s) if s.get("card") else f"[{s['n']}] {s['title']} ({_year(s)}): {s.get('key_sentences', '')}"


def broad_material(kind: str, sources: list[dict], themes: list[dict], written: list, theme=None) -> str:
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
        return "\n\n".join(card_text(s) for s in deep) + so_far
    if kind == "setup":
        return "\n\n".join(card_text(s, ("method", "setup", "findings")) for s in deep) + so_far
    if kind == "limits":
        return "\n\n".join(card_text(s, ("findings", "limitations")) for s in deep) + so_far
    return so_far.strip()


def write_broad(doc_type: str, sources: list[dict], themes: list[dict], topic: str, lang: str,
                model: str) -> tuple[str, int, list[dict]]:
    spec, language, n = BROAD[doc_type], LANGUAGES[lang], len(sources)
    label = f"{spec['label']} on {topic}"
    plan = []
    for sec in spec["sections"]:
        if sec == THEMES:
            plan += [{"key": f"theme:{t['name']}", "heading": spec["theme_prefix"][lang] + t["name"], "kind": "theme",
                      "task": spec["theme_task"], "tokens": 2500, "last": False, "theme": t} for t in themes]
        else:
            plan.append({**sec, "heading": sec["title"][lang], "theme": None})
    outline = "\n".join(f"## {p['heading']}" for p in plan)
    texts, written, report, used = {}, [], [], 0
    for item in [p for p in plan if not p["last"]] + [p for p in plan if p["last"]]:
        prompt = SECTION_PROMPT.format(
            label=label, language=language, outline=outline, written=summary_of(written),
            target=f'the section "{item["heading"]}"', task=item["task"].format(n=n, k=len(themes)),
            context_rule="", material=broad_material(item["kind"], sources, themes, written, item["theme"]),
            style=STYLE, n=n, extra=BROAD_RULES)
        text, tokens, truncated = _call(prompt, model, item["tokens"])
        used += tokens
        report.append({"section": item["key"], "source": None, "tokens": tokens, "truncated": truncated})
        texts[item["heading"]] = text.strip()
        written.append((item["heading"], texts[item["heading"]]))
    title, tokens, _ = _call(TITLE_PROMPT.format(language=language, label=label, written=summary_of(written)),
                             model, 300)
    used += tokens
    title = title.strip().strip('"').strip("#").strip() or topic
    body = "\n\n".join(f"## {p['heading']}\n\n{texts[p['heading']]}" for p in plan)
    return f"# {title}\n\n{body}\n", used, report
