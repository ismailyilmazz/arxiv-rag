import pytest

from core.arxiv_ids import abs_url, normalize_id


@pytest.mark.parametrize("raw, expected", [
    ("2310.01234", "2310.01234"),               # yeni tip
    ("0704.0001", "0704.0001"),                 # yeni tipin ilk dönemi (4 haneli)
    ("hep-th/9901001", "hep-th/9901001"),       # eski tip
    ("math.GT/0309136", "math.GT/0309136"),     # eski tip, alt kategorili
    ("oai:arXiv.org:2310.01234", "2310.01234"), # OAI-PMH biçimi
    ("arXiv:2310.01234v2", "2310.01234"),       # sürüm ekli
    ("https://arxiv.org/abs/hep-th/9901001v3", "hep-th/9901001"),
])
def test_normalize_id(raw, expected):
    assert normalize_id(raw) == expected


@pytest.mark.parametrize("bad", ["", "cs-9308101v1", "abs-0711.2058v1", "hello"])
def test_normalize_id_rejects_unknown(bad):
    # Eski projedeki CSV'nin bozuk kimlikleri artık sessizce kırık link üretmiyor, hata veriyor.
    with pytest.raises(ValueError):
        normalize_id(bad)


def test_abs_url():
    assert abs_url("hep-th/9901001") == "https://arxiv.org/abs/hep-th/9901001"
