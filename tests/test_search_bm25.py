import pytest

from core.search_bm25 import search, to_fts_query


def test_query_is_sanitized():
    # Tırnak, parantez, iki nokta ve AND/NOT gibi FTS5 operatörleri hata vermemeli
    q = to_fts_query('Graph "neural" (networks) AND traffic: NOT forecasting*')
    assert q == '"graph" OR "neural" OR "networks" OR "traffic" OR "forecasting"'


def test_stopwords_and_duplicates_removed():
    assert to_fts_query("the theory of the black hole hole") == '"theory" OR "black" OR "hole"'


def test_empty_query():
    assert to_fts_query("the of and") == ""


def test_finds_right_paper(small_db):
    conn, _ = small_db
    results = search(conn, "robot grasping", k=3)
    assert results[0][0] == "2310.00002"
    assert results[0][1] > 0                      # skor pozitif, büyük olan daha alakalı


def test_plural_and_singular_match(small_db):
    conn, _ = small_db
    assert search(conn, "neural network", k=1)[0][0] == "2310.00001"


@pytest.mark.parametrize("text", ["", "the of", '"""', "NOT AND OR", "çikolatalı kek tarifi"])
def test_odd_inputs_do_not_crash(small_db, text):
    conn, _ = small_db
    assert isinstance(search(conn, text), list)
