import re

from core import llm

LANGUAGES = {"en": "English", "tr": "Turkish"}
STYLE = ("Write complete sentences, each with an explicit subject and a finite verb; in Turkish every sentence ends "
         "with its predicate. Prefer flowing paragraphs; use bullet points only for short enumerations and write "
         "each bullet as a full sentence.")
TOKEN_BUDGET = 7600


def section(key, en, tr, material, task, max_tokens, last=False):
    return {"key": key, "title": {"en": en, "tr": tr}, "material": material, "task": task,
            "max_tokens": max_tokens, "last": last}


DOC_TYPES = {
    "survey": {"label": "literature survey", "sections": [
        section("intro", "Introduction", "Giriş", ["brief"],
                "Motivate the topic, define its scope, state what this survey covers and outline the following "
                "sections.", 1500),
        section("background", "Background", "Arka Plan", ["brief"],
                "Explain the concepts, terminology and earlier ideas a reader needs before the individual "
                "approaches.", 1500),
        section("approaches", "Approaches", "Yaklaşımlar", ["source"],
                "Describe this source in depth: the problem it addresses, how its method works including the key "
                "design choices, its experimental set-up, its results with their exact numbers and its stated "
                "limitations. Use the excerpt for details the card does not contain.", 2500),
        section("comparison", "Comparison and Discussion", "Karşılaştırma ve Tartışma", ["full"],
                "Compare the sources along design, training and data, efficiency, results and limitations. Use a "
                "level-3 heading for each dimension and point out agreements, disagreements and trade-offs.", 2500),
        section("open", "Open Problems", "Açık Problemler", ["limits"],
                "Derive open problems and research directions from the limitations and gaps of the sources.", 1500),
        section("conclusion", "Conclusion", "Sonuç", ["written"],
                "Summarize the main insights of this survey without introducing new claims.", 1000),
    ]},
    "proposal": {"label": "research proposal", "sections": [
        section("abstract", "Abstract", "Özet", ["written"],
                "Write the abstract of the proposal: the problem, the gap, the proposed approach, the evaluation and "
                "the expected contributions, in one or two paragraphs.", 800, last=True),
        section("problem", "Problem and Motivation", "Problem ve Motivasyon", ["brief", "limits"],
                "State the problem and why it matters; use the sources to show what is already solved and what "
                "remains open.", 1500),
        section("related", "Related Work", "İlgili Çalışmalar", ["source"],
                "Discuss this source as related work: its method, set-up and results with their numbers, and the "
                "limitation that matters for this proposal.", 2000),
        section("questions", "Research Questions", "Araştırma Soruları", ["limits", "written"],
                "Formulate three to five research questions that follow from the gaps, and explain each one.", 1200),
        section("method", "Proposed Method", "Önerilen Yöntem", ["full", "written"],
                "Describe the proposed new method as a plan: its components, data, training and how it answers each "
                "research question. Never present results.", 2500),
        section("evaluation", "Evaluation Plan", "Değerlendirme Planı", ["full", "written"],
                "Describe the evaluation plan: data sets, baselines taken from the sources, metrics and ablations.",
                1800),
        section("contributions", "Expected Contributions", "Beklenen Katkılar", ["written"],
                "State the expected contributions of the proposed work.", 1000),
        section("risks", "Risks and Limitations", "Riskler ve Sınırlar", ["written", "limits"],
                "Discuss the risks and limitations of the plan and how they will be mitigated.", 1200),
    ]},
    "synthesis": {"label": "synthesis report", "sections": [
        section("summary", "Summary", "Özet", ["written"],
                "Summarize what these sources show when taken together, in one or two paragraphs.", 800, last=True),
        section("findings", "Key Findings", "Temel Bulgular", ["source"],
                "Present the findings of this source with their numbers and conditions, and the evidence behind "
                "them.", 2000),
        section("agreements", "Agreements and Differences", "Ortak Noktalar ve Farklar", ["full"],
                "Compare the sources point by point: where they agree, where they differ and why.", 2000),
        section("implications", "Implications", "Çıkarımlar", ["full", "written"],
                "Discuss the implications of these findings for research and practice.", 1500),
        section("conclusion", "Conclusion", "Sonuç", ["written"],
                "Conclude the report without introducing new claims.", 800),
    ]},
}

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
- Length: as long as the material supports for this part, and no longer. Never pad.
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


def material_for(spec: dict, sources: list[dict], written: list[tuple[str, str]], source=None) -> str:
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
    return "\n\n".join(parts)


def _call(prompt: str, model: str, max_tokens: int) -> tuple[str, int, bool]:
    budget = max(400, min(max_tokens, TOKEN_BUDGET - len(prompt) // 3))
    text, tokens, finish = llm.complete(prompt, model=model, max_tokens=budget)
    return text, tokens, finish == "length"


def write_document(sources: list[dict], doc_type: str, lang: str, model: str) -> tuple[str, int, list[dict]]:
    spec, language = DOC_TYPES[doc_type], LANGUAGES[lang]
    outline = "\n".join(f"## {s['title'][lang]}" for s in spec["sections"])
    order = [s for s in spec["sections"] if not s["last"]] + [s for s in spec["sections"] if s["last"]]
    written, parts, report, total = [], {}, [], 0

    for sec in order:
        heading = sec["title"][lang]
        targets = [(f"the subsection about source [{s['n']}] in the section \"{heading}\"", s)
                   for s in sources] if sec["material"] == ["source"] else [(f"the section \"{heading}\"", None)]
        chunks = []
        for target, source in targets:
            prompt = SECTION_PROMPT.format(
                label=spec["label"], language=language, outline=outline, written=summary_of(written),
                target=target, task=sec["task"], material=material_for(sec, sources, written, source),
                style=STYLE, n=len(sources))
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
