import re

_NEW_STYLE = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_STYLE = re.compile(r"^[a-z][a-z\-]*(\.[A-Z]{2})?/\d{7}$")
_VERSION = re.compile(r"v\d+$")
_PREFIXES = (
    "oai:arxiv.org:",
    "arxiv:",
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
)


def normalize_id(raw: str) -> str:
    s = raw.strip()
    lowered = s.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            s = s[len(prefix):]
            break
    s = _VERSION.sub("", s)
    if not (_NEW_STYLE.match(s) or _OLD_STYLE.match(s)):
        raise ValueError(f"Tanınmayan arXiv kimliği: {raw!r}")
    return s


def abs_url(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{normalize_id(paper_id)}"
