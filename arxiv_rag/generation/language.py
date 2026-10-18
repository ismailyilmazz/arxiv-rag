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


def paragraph_is_english(paragraph: str) -> bool:
    words = [w.lower() for w in WORD.findall(paragraph)]
    english = sum(w in EN_WORDS for w in words)
    turkish = sum(w in TR_WORDS or any(c in TR_CHARS for c in w) for w in words)
    return english >= 3 and english > 2 * turkish


def prose_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        lines = [line for line in block.splitlines() if line.strip() and not line.lstrip().startswith(("|", "#"))]
        paragraph = " ".join(lines).strip()
        if len(paragraph.split()) >= 8:
            paragraphs.append(paragraph)
    return paragraphs


def english_share(text: str) -> float:
    paragraphs = prose_paragraphs(text)
    return sum(map(paragraph_is_english, paragraphs)) / len(paragraphs) if paragraphs else 0.0


def audit(text: str) -> dict:
    paragraphs = prose_paragraphs(text)
    english = sum(map(paragraph_is_english, paragraphs))
    return {
        "paragraphs": len(paragraphs),
        "english_paragraphs": english,
        "english_share": round(english / len(paragraphs), 3) if paragraphs else 0.0,
    }
