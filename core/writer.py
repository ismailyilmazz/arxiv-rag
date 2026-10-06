from core import llm

LANGUAGES = {"en": "English", "tr": "Turkish"}
DOC_TYPES = {
    "survey": {
        "label": "literature survey",
        "sections": {
            "en": ["Introduction", "Background", "Approaches", "Comparison and Discussion", "Open Problems", "Conclusion"],
            "tr": ["Giriş", "Arka Plan", "Yaklaşımlar", "Karşılaştırma ve Tartışma", "Açık Problemler", "Sonuç"],
        },
        "guide": "Organize the work thematically, compare the approaches explicitly and show where the sources "
                 "agree or disagree.",
    },
    "proposal": {
        "label": "research proposal",
        "sections": {
            "en": ["Abstract", "Problem and Motivation", "Related Work", "Research Questions", "Proposed Method",
                   "Evaluation Plan", "Expected Contributions", "Risks and Limitations"],
            "tr": ["Özet", "Problem ve Motivasyon", "İlgili Çalışmalar", "Araştırma Soruları", "Önerilen Yöntem",
                   "Değerlendirme Planı", "Beklenen Katkılar", "Riskler ve Sınırlar"],
        },
        "guide": "Use the sources to establish the gap. The proposed method must be new and is described as a plan, "
                 "never as an obtained result.",
    },
    "synthesis": {
        "label": "synthesis report",
        "sections": {
            "en": ["Summary", "Key Findings", "Agreements and Differences", "Implications", "Conclusion"],
            "tr": ["Özet", "Temel Bulgular", "Ortak Noktalar ve Farklar", "Çıkarımlar", "Sonuç"],
        },
        "guide": "Focus on what a reader needs to know from these sources taken together.",
    },
}

PROMPT = """Write a {label} in {language}, based only on the numbered sources below.

Sources:
{sources}

Structure: start with a level-1 heading containing a title you write, then use exactly these level-2 headings in this order:
{sections}

Rules:
- {guide}
- Every paragraph and every bullet point that describes, compares or relies on the sources must contain at least one citation in square brackets, such as [1] or [1, 3]. Use only the numbers 1 to {n}.
- When you mention a source by name, put its number next to it, for example "Switch Transformers [2]".
- Do not invent results, numbers, datasets or methods that are not in the sources.
- Do not write a references or bibliography section; it is added automatically.
- Write in {language}. Length: as long as the sources support, and no longer. Treat every source in depth (method, set-up, results with their numbers) and compare the sources explicitly. When the material is exhausted, stop. Never pad with repetition, generic statements about the field or guesses about what the sources might contain.
- Keep technical terms that are normally used in English as they are.
"""


def source_block(sources: list[dict]) -> str:
    blocks = []
    for s in sources:
        c = s["card"]
        year = (s.get("published") or "")[:4] or "n.d."
        lines = [f"[{s['n']}] {s['title']} ({year})"]
        lines += [f"{key.capitalize()}: {c[key]}" for key in ("problem", "method", "setup", "findings", "limitations")
                  if c.get(key)]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


TOKEN_BUDGET = 7600


def build_prompt(sources: list[dict], doc_type: str, lang: str) -> str:
    spec = DOC_TYPES[doc_type]
    return PROMPT.format(label=spec["label"], language=LANGUAGES[lang], sources=source_block(sources),
                         sections="\n".join(f"## {h}" for h in spec["sections"][lang]),
                         guide=spec["guide"], n=len(sources))


def output_budget(prompt: str) -> int:
    return max(1500, TOKEN_BUDGET - len(prompt) // 3)


def write(sources: list[dict], doc_type: str, lang: str, model: str) -> tuple[str, int, bool, int]:
    prompt = build_prompt(sources, doc_type, lang)
    budget = output_budget(prompt)
    text, tokens, finish = llm.complete(prompt, model=model, max_tokens=budget)
    return text, tokens, finish == "length", budget
