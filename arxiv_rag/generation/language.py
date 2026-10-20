import re

TR_CHARS = set("çğıöşüÇĞİÖŞÜ")
TR_WORDS = {
    "ve",
    "bir",
    "bu",
    "ile",
    "için",
    "olarak",
    "da",
    "de",
    "olan",
    "gibi",
    "daha",
    "ancak",
    "çok",
    "ise",
    "veya",
    "her",
    "ayrıca",
    "üzerinde",
    "tarafından",
    "göre",
    "kadar",
    "sonra",
    "önce",
    "arasında",
    "değil",
    "şekilde",
    "ki",
}
EN_WORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "is",
    "that",
    "for",
    "with",
    "as",
    "are",
    "by",
    "on",
    "this",
    "which",
    "from",
    "an",
    "be",
    "these",
    "their",
    "while",
    "such",
    "can",
    "has",
    "have",
    "was",
    "were",
    "its",
    "into",
    "than",
}
WORD = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ]+")


EN_SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ing", "ance", "ence", "ship", "ics", "ous", "ive", "ed", "ies")
STOP = EN_WORDS | {
    "a",
    "also",
    "all",
    "both",
    "but",
    "each",
    "more",
    "most",
    "not",
    "or",
    "other",
    "only",
    "such",
    "them",
    "they",
    "those",
    "through",
    "under",
    "using",
    "used",
    "uses",
    "use",
    "when",
    "where",
    "whether",
    "within",
    "without",
    "work",
    "works",
    "show",
    "shows",
    "shown",
    "however",
    "first",
    "second",
    "third",
    "new",
    "two",
    "three",
    "several",
    "many",
    "while",
    "between",
    "across",
    "based",
    "different",
    "further",
    "make",
    "makes",
    "provide",
    "provides",
    "study",
    "studies",
    "approach",
    "approaches",
    "result",
    "results",
    "paper",
    "papers",
    "theme",
    "themes",
    "section",
    "review",
    "rather",
    "even",
    "then",
    "there",
    "thus",
    "therefore",
    "because",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "been",
    "being",
    "does",
    "do",
    "did",
    "it",
    "we",
    "our",
    "how",
    "what",
    "who",
    "why",
    "into",
    "over",
    "about",
    "after",
    "before",
    "against",
    "per",
    "via",
}
SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-"})
TERM = re.compile(r"[A-Za-z][A-Za-z-]*[A-Za-z]")
MARKED = re.compile(r"\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[[\d,\s\u2013-]+\]|\u201c[^\u201d\n]+\u201d|\"[^\"\n]+\"")
SPAN = re.compile(r"\*\*([^*]+)\*\*|\*([^*\s][^*]*)\*|\u201c([^\u201d\n]+)\u201d|\"([^\"\n]+)\"")


def normalize(text: str) -> str:
    text = re.sub(r"\[[\d,\s\u2013-]+\]", " ", text.translate(HYPHENS)).lower()
    return " ".join(re.sub(r"[\"'\u201c\u201d\u2018\u2019]", " ", text).split()).strip(" .,:;")


def paragraph_is_english(paragraph: str) -> bool:
    words = [w.lower() for w in WORD.findall(paragraph)]
    english = sum(w in EN_WORDS for w in words)
    turkish = sum(w in TR_WORDS or any(c in TR_CHARS for c in w) for w in words)
    return english >= 3 and english > 2 * turkish


def _english_signal(word: str) -> bool:
    return word in EN_WORDS or any(c in "qwx" for c in word) or (len(word) > 4 and word.endswith(EN_SUFFIXES))


def english_words(text: str) -> list[str]:
    plain = re.sub(r"['\u2019][A-Za-zçğıöşüÇĞİÖŞÜ]+", "", MARKED.sub(" ", text.translate(HYPHENS)))
    return [
        w
        for w in WORD.findall(plain)
        if not w.isupper() and not any(c in TR_CHARS for c in w) and _english_signal(w.lower())
    ]


def text_is_english(text: str) -> bool:
    words = [w for w in WORD.findall(text) if not w.isupper()]
    if not words:
        return False
    lower = [w.lower() for w in words]
    english = sum(w in EN_WORDS for w in lower)
    turkish = sum(w.lower() in TR_WORDS or any(c in TR_CHARS for c in w) for w in words)
    if english >= 3 and english > 2 * turkish:
        return True
    return turkish == 0 and len(words) >= 2 and any(map(_english_signal, lower))


def table_text(lines: list[str]) -> str:
    cells = []
    for line in lines[1:]:
        if SEPARATOR.match(line):
            continue
        cells += [MARKED.sub(" ", c).strip() for c in line.strip().strip("|").split("|")][1:]
    return " | ".join(cells)


def units(text: str) -> list[tuple[str, str]]:
    out = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        table = [line for line in lines if line.startswith("|")]
        if table:
            out.append(("table", table_text(table)))
            out.append(("line", MARKED.sub(" ", " ".join(table[0].strip().strip("|").split("|"))).strip()))
        out += [("line", MARKED.sub(" ", line.lstrip("#")).strip()) for line in lines if line.startswith("#")]
        prose = " ".join(MARKED.sub(" ", " ".join(line for line in lines if not line.startswith(("|", "#")))).split())
        if prose:
            out.append(("paragraph" if len(prose.split()) >= 8 else "line", prose))
    return [(kind, text) for kind, text in out if text]


def unit_is_english(kind: str, text: str) -> bool:
    return paragraph_is_english(text) if kind == "paragraph" else text_is_english(text)


def block_is_english(block: str) -> bool:
    return any(unit_is_english(kind, text) for kind, text in units(block))


def has_english(text: str) -> bool:
    return block_is_english(text)


def prose_paragraphs(text: str) -> list[str]:
    return [t for kind, t in units(text) if kind == "paragraph"]


def english_share(text: str) -> float:
    paragraphs = prose_paragraphs(text)
    return sum(map(paragraph_is_english, paragraphs)) / len(paragraphs) if paragraphs else 0.0


def audit(text: str) -> dict:
    found = units(text)
    counts = {kind: [unit_is_english(k, t) for k, t in found if k == kind] for kind in ("paragraph", "table", "line")}
    paragraphs, english = len(counts["paragraph"]), sum(counts["paragraph"])
    return {
        "paragraphs": paragraphs,
        "english_paragraphs": english,
        "english_share": round(english / paragraphs, 3) if paragraphs else 0.0,
        "tables": len(counts["table"]),
        "english_tables": sum(counts["table"]),
        "lines": len(counts["line"]),
        "english_lines": sum(counts["line"]),
        "examples": [t[:80] for k, t in found if k != "paragraph" and unit_is_english(k, t)][:5],
    }


def frequent_terms(text: str, unigrams: int = 40, bigrams: int = 25, min_count: int = 2) -> list[str]:
    text = _translatable(text)
    ones, twos = {}, {}
    for sentence in re.split(r"[.!?;:()|\n]+", text):
        words = [w.lower() for w in TERM.findall(sentence) if not w.isupper()]
        for w in words:
            if w not in STOP and len(w) >= 5:
                ones[w] = ones.get(w, 0) + 1
        for a, b in zip(words, words[1:]):
            if a not in STOP and b not in STOP and len(a) >= 3 and len(b) >= 3:
                twos[f"{a} {b}"] = twos.get(f"{a} {b}", 0) + 1

    def top(counts, n):
        return [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_count][:n]

    return top(twos, bigrams) + top(ones, unigrams)


def _translatable(text: str) -> str:
    kept = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        table = [line for line in lines if line.startswith("|")]
        if table:
            kept.append(table_text(table))
        kept += [line.lstrip("#").strip() for line in lines if not line.startswith("|")]
    return MARKED.sub(" ", "\n".join(kept).translate(HYPHENS))


def term_audit(text: str, glossary: dict) -> dict:
    plain = _translatable(text).lower()
    left = {}
    for en, target in glossary.items():
        term = en.translate(HYPHENS).lower().strip()
        if not term or term == target.translate(HYPHENS).lower().strip() or term.upper() == en:
            continue
        n = len(re.findall(rf"(?<![a-z]){re.escape(term)}(?![a-z])", plain))
        if n:
            left[en] = n
    return {
        "terms": len(glossary),
        "english_left": sum(left.values()),
        "examples": dict(sorted(left.items(), key=lambda kv: -kv[1])[:5]),
    }
