from core import llm

LANGUAGES = {"en": "English", "tr": "Turkish"}
DOC_TYPES = {
    "survey": {
        "label": "literature survey",
        "words": 1600,
        "sections": {
            "en": ["Introduction", "Background", "Approaches", "Comparison and Discussion", "Open Problems", "Conclusion"],
            "tr": ["Giriş", "Arka Plan", "Yaklaşımlar", "Karşılaştırma ve Tartışma", "Açık Problemler", "Sonuç"],
        },
        "guide": "Organize the work thematically, compare the approaches explicitly and show where the sources "
                 "agree or disagree.",
    },
    "proposal": {
        "label": "research proposal",
        "words": 1400,
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
        "label": "short synthesis report",
        "words": 900,
        "sections": {
            "en": ["Summary", "Key Findings", "Agreements and Differences", "Implications", "Conclusion"],
            "tr": ["Özet", "Temel Bulgular", "Ortak Noktalar ve Farklar", "Çıkarımlar", "Sonuç"],
        },
        "guide": "Be concise and focus on what a reader needs to know from these sources taken together.",
    },
}

PROMPT = """Write a {label} in {language}, based only on the numbered sources below.

Sources:
{sources}

Structure: start with a level-1 heading containing a title you write, then use exactly these level-2 headings in this order:
{sections}

Rules:
- {guide}
- Support every statement about a source with a citation in square brackets, such as [1] or [1, 3]. Use only the numbers 1 to {n}.
- Do not invent results, numbers, datasets or methods that are not in the sources.
- Do not write a references or bibliography section; it is added automatically.
- Write about {words} words in {language}. Keep technical terms that are normally used in English as they are.
"""


def source_block(sources: list[dict]) -> str:
    blocks = []
    for s in sources:
        c = s["card"]
        year = (s.get("published") or "")[:4] or "n.d."
        blocks.append(f"[{s['n']}] {s['title']} ({year})\n"
                      f"Problem: {c['problem']}\nMethod: {c['method']}\n"
                      f"Findings: {c['findings']}\nLimitations: {c['limitations']}")
    return "\n\n".join(blocks)


def build_prompt(sources: list[dict], doc_type: str, lang: str) -> str:
    spec = DOC_TYPES[doc_type]
    return PROMPT.format(label=spec["label"], language=LANGUAGES[lang], sources=source_block(sources),
                         sections="\n".join(f"## {h}" for h in spec["sections"][lang]),
                         guide=spec["guide"], n=len(sources), words=spec["words"])


def write(sources: list[dict], doc_type: str, lang: str, model: str) -> tuple[str, int]:
    words = DOC_TYPES[doc_type]["words"]
    return llm.complete(build_prompt(sources, doc_type, lang), model=model, max_tokens=min(5500, words * 3))
