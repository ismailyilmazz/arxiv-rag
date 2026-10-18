import pytest

from arxiv_rag.storage.tex import clean_tex


@pytest.mark.parametrize(
    "raw, clean",
    [
        (
            r"B\"{u}y\"{u}k dil modellerinin T\"{u}rk\c{c}e verisetleri ile e\u{g}itilmesi ve ince ayarlanmas\i",
            "Büyük dil modellerinin Türkçe verisetleri ile eğitilmesi ve ince ayarlanması",
        ),
        (r"M\"uller and G\"{o}del", "Müller and Gödel"),
        (r"{\'E}cole Polytechnique", "École Polytechnique"),
        (r"Erd\H{o}s, \v{S}koda, Fran\c cois", "Erdős, Škoda, François"),
        (r"na\"{\i}ve Stra\ss e, \O resund", "naïve Straße, Øresund"),
        (r"\c{S}ahin, \u{G}\"{u}ne\c{s}, I\c{s}{\i}k", "Şahin, Ğüneş, Işık"),
    ],
)
def test_accents_become_letters(raw, clean):
    assert clean_tex(raw) == clean


@pytest.mark.parametrize(
    "text",
    [
        "Attention Is All You Need",
        r"An $O(n \log n)$ algorithm with $\alpha$-stable noise",
        r"The \textit{vec} trick",
        "",
    ],
)
def test_other_text_is_untouched(text):
    assert clean_tex(text) == text


def test_none_becomes_empty():
    assert clean_tex(None) == ""
