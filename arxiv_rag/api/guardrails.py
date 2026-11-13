import re
from typing import Optional

ABUSE = frozenset({"profanity", "injection", "spam"})

MESSAGES = {
    "empty": "Kutuda anlamlı bir kelime yok. Bir araştırma konusu yaz.",
    "gibberish": "Bu girdi rastgele tuşlara benziyor. Bir araştırma konusu yaz.",
    "profanity": "Bu girdi uygunsuz ifadeler içeriyor; işlenmedi.",
    "injection": "Bu girdi sisteme yönelik talimatlar içeriyor; işlenmedi. Sadece araştırma konusunu yaz.",
    "spam": "Bu girdi bağlantı, e-posta ya da reklam içeriyor; işlenmedi.",
}

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_VOWELS = set("aeıioöuüyâîû")
_ROWS = ("qwertyuıopğü", "asdfghjklşi", "zxcvbnmöç", "qwertyuiop", "asdfghjkl", "zxcvbnm")

_WORDS = frozenset(
    {
        "amk",
        "amq",
        "amına",
        "amina",
        "amcık",
        "amcik",
        "orospu",
        "orosbu",
        "yarrak",
        "yarak",
        "siktir",
        "sikerim",
        "sikeyim",
        "piç",
        "göt",
        "götveren",
        "gotveren",
        "ibne",
        "pezevenk",
        "gavat",
        "kahpe",
        "yavşak",
        "yavsak",
        "kaltak",
        "şerefsiz",
        "serefsiz",
        "fuck",
        "fck",
        "fucking",
        "fucked",
        "fucker",
        "shit",
        "shitty",
        "bullshit",
        "bitch",
        "asshole",
        "cunt",
        "dickhead",
        "slut",
        "whore",
    }
)
_STEMS = ("orospu", "yarrak", "yarağ", "amcık", "amcik", "siktir", "pezevenk", "götveren", "bitch", "whore", "asshole")
_INSIDE = ("fuck",)
_RAW_ONLY = frozenset({"nigger", "nigga", "faggot"})

_INJECTION = [
    re.compile(p, re.I)
    for p in (
        r"\b(ignore|disregard|forget|override|bypass)\b[^.]{0,20}\b(previous|prior|above|earlier|preceding|your)\b"
        r"[^.]{0,20}\b(instructions?|prompts?|rules?|directions?|guidelines?)\b",
        r"\bif you are (an? )?(ai|llm|language model|assistant|chat ?gpt|gpt|gemini|claude|bot|model)\b",
        r"\b(you are|you're|pretend (to be|you are)|from now on,? you)\b[^.]{0,25}"
        r"\b(an? )?(ai|assistant|chat ?gpt|gpt|gemini|claude|dan|jailbroken|unrestricted|unfiltered)\b",
        r"\b(write|print|output|say|repeat|insert|start|begin|end|respond|reply)\b[^.]{0,40}"
        r"\b(the (exact )?(sentence|phrase|string)|verbatim)\b",
        r"\bdo anything now\b|\bdeveloper mode\b",
        r"\b(önceki|yukarıdaki|tüm|bütün|verilen|bu)\b[^.]{0,30}\b(talimat|komut|kural|yönerge)\w*[^.]{0,30}"
        r"\b(yok say|unut|görmezden gel)\b",
        r"\b(talimat|komut|kural|yönerge)\w*\s+(yok say|unut|görmezden gel)\b",
        r"(zeka|zekâ|asistan|gemini|chatgpt|claude|gpt|bot|model)(ysan|ysen|isen|ısan|san|sen)\b",
        r"\bsen (artık|bundan sonra|şimdi)\b[^.]{0,30}\b(yapay zek|asistan|model|bot|rol)",
        r"\b(şu|bu|aşağıdaki)\s+(cümle|kelime|ifade|metin|metni|yazı)\w*\s+[^.]{0,30}\b(yaz|ekle|başla)\b",
        r"\b(girişe|girişin başına|başa|sona|her bölüme|belgeye)\b[^.]{0,40}\b(yaz|ekle)\b",
    )
]
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|'[^']*'|‘[^’]*’")

_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_ALLOWED_HOSTS = re.compile(r"^(https?://)?(www\.)?(arxiv\.org|doi\.org)(/|$)", re.I)
_SPAM = [
    re.compile(p, re.I)
    for p in (
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        r"\b(bc1[a-z0-9]{25,60}|[13][a-km-zA-HJ-NP-Z1-9]{25,34}|0x[a-fA-F0-9]{40})\b",
        r"\+\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b",
        r"\b0\s?5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b",
        r"\b(click here|buy now|promo code|discount code|free money|t\.me/|tıkla|tıklayın|hemen üye ol|hemen kayıt ol|"
        r"promosyon kodu|bonus kodu|deneme bonusu|bedava bonus)",
    )
]


def _tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _collapse(word: str) -> str:
    return re.sub(r"(.)\1+", r"\1", word)


def _tokens(form: str) -> list[str]:
    out, run = [], []
    for raw in form.split() + [""]:
        word = "".join(ch for ch in raw.translate(_LEET) if ch.isalpha())
        if len(word) == 1:
            run.append(word)
            continue
        if len(run) >= 3:
            out.append("".join(run))
        run = []
        if word:
            out.append(word)
    return out


def _profane(text: str) -> bool:
    words = {_collapse(w) for w in _WORDS}
    stems = tuple(_collapse(s) for s in _STEMS)
    for form in {_tr_lower(text), text.lower()}:
        for word in _tokens(form):
            short = _collapse(word)
            if word in _RAW_ONLY or short in words or short.startswith(stems) or any(i in short for i in _INSIDE):
                return True
    return False


def _injection(text: str) -> bool:
    bare = _QUOTED.sub(" ", text)
    return any(p.search(form) for form in {_tr_lower(bare), bare.lower()} for p in _INJECTION)


def _spam(text: str) -> bool:
    if any(not _ALLOWED_HOSTS.match(m.group(0)) for m in _URL.finditer(text)):
        return True
    return any(p.search(text) for p in _SPAM)


def _mash(word: str) -> bool:
    if re.search(r"(.)\1{3,}", word) or re.fullmatch(r"(\w{1,4})\1{2,}", word):
        return True
    if any(word[i : i + 5] in row or word[i : i + 5][::-1] in row for row in _ROWS for i in range(len(word) - 4)):
        return True
    return len(word) >= 6 and not _VOWELS & set(word)


def _gibberish(text: str) -> bool:
    words = [w for w in re.findall(r"[^\W\d_]+", text) if len(w) >= 3]
    if not words or any(w.isupper() for w in words):
        return False
    return all(_mash(_tr_lower(w)) for w in words)


def screen(text: str) -> Optional[str]:
    if len(re.findall(r"[^\W\d_]", text or "")) < 2:
        return "empty"
    if _profane(text):
        return "profanity"
    if _injection(text):
        return "injection"
    if _spam(text):
        return "spam"
    if _gibberish(text):
        return "gibberish"
    return None
