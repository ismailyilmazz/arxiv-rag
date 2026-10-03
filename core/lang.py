import re

_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TURKISH_WORDS = {
    "ve", "ile", "için", "bir", "bu", "nasıl", "ne", "olan", "olmayan", "gibi", "üzerine", "hakkında",
    "makale", "makalesi", "orijinal", "yöntemi", "yöntem", "modeli", "modelleri", "kelime", "veri",
    "verisetleri", "eğitme", "eğitimi", "tespiti", "algoritması", "mekanizması", "arama", "üreten",
}
_TURKISH_SUFFIXES = ("leri", "ları", "ması", "mesi", "lık", "lik", "sını", "sini", "ndan", "nden")
_WORD = re.compile(r"\w+", re.UNICODE)


def is_turkish(text: str) -> bool:
    if any(ch in _TURKISH_CHARS for ch in text):
        return True
    words = _WORD.findall(text.lower())
    return any(w in _TURKISH_WORDS or (len(w) > 6 and w.endswith(_TURKISH_SUFFIXES)) for w in words)
