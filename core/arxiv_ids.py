"""arXiv kimlikleri farklı kaynaklardan farklı biçimlerde gelir.

    Kaggle snapshot : 2310.01234         hep-th/9901001
    OAI-PMH         : oai:arXiv.org:2310.01234
    Kullanıcı       : arXiv:2310.01234v2  https://arxiv.org/abs/2310.01234

Hepsini tek biçime indiriyoruz: sürüm eki olmayan çıplak kimlik.
"""
import re

_NEW_STYLE = re.compile(r"^\d{4}\.\d{4,5}$")                  # 2007 ve sonrası
_OLD_STYLE = re.compile(r"^[a-z][a-z\-]*(\.[A-Z]{2})?/\d{7}$")  # 2007 öncesi: hep-th/9901001, math.GT/0309136
_VERSION = re.compile(r"v\d+$")
_PREFIXES = (
    "oai:arxiv.org:",
    "arxiv:",
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
)


def normalize_id(raw: str) -> str:
    """Kimliği sürüm eki olmayan çıplak biçime çevirir. Tanınmazsa ValueError verir."""
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
    """Makalenin arXiv özet sayfası. arXiv kullanım koşulları PDF yerine bu sayfaya yönlendirmeyi önerir."""
    return f"https://arxiv.org/abs/{normalize_id(paper_id)}"
