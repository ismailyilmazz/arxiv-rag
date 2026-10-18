import re
import unicodedata
from typing import Optional

ACCENTS = {
    '"': "\u0308",
    "'": "\u0301",
    "`": "\u0300",
    "^": "\u0302",
    "~": "\u0303",
    "=": "\u0304",
    ".": "\u0307",
    "u": "\u0306",
    "v": "\u030c",
    "H": "\u030b",
    "c": "\u0327",
    "k": "\u0328",
    "r": "\u030a",
    "d": "\u0323",
}
LETTERS = {
    "ss": "ß",
    "ae": "æ",
    "AE": "Æ",
    "oe": "œ",
    "OE": "Œ",
    "aa": "å",
    "AA": "Å",
    "i": "ı",
    "j": "ȷ",
    "o": "ø",
    "O": "Ø",
    "l": "ł",
    "L": "Ł",
}
ARG = r"(?:\{\s*(\\[ij]|[A-Za-z])\s*\}|(\\[ij](?![A-Za-z])|[A-Za-z]))"
SYMBOL_ACCENT = re.compile(r"\\([\"'`^~=.])\s*" + ARG)
LETTER_ACCENT = re.compile(r"\\([uvHckrd])(?![A-Za-z])(?:\s*\{\s*(\\[ij]|[A-Za-z])\s*\}|\s+([A-Za-z]))")
LETTER_MACRO = re.compile(r"\\(ss|ae|AE|oe|OE|aa|AA|[ijoOlL])(?![A-Za-z])(?:\{\}|\s)?")
BRACED_LETTER = re.compile(r"\{([^\W\d_])\}")


def _accent(match: re.Match) -> str:
    base = match.group(2) or match.group(3)
    base = base[1] if base.startswith("\\") else base
    return base + ACCENTS[match.group(1)]


def clean_tex(text: Optional[str]) -> str:
    if not text or "\\" not in text and "{" not in text:
        return text or ""
    out = SYMBOL_ACCENT.sub(_accent, text)
    out = LETTER_ACCENT.sub(_accent, out)
    out = LETTER_MACRO.sub(lambda m: LETTERS[m.group(1)], out)
    out = unicodedata.normalize("NFC", out)
    return BRACED_LETTER.sub(r"\1", out)
