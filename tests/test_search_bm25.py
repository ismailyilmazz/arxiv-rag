import pytest

from core.search_bm25 import search, to_fts_query


def test_query_is_sanitized():
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
    assert results[0][1] > 0


def test_plural_and_singular_match(small_db):
    conn, _ = small_db
    assert search(conn, "neural network", k=1)[0][0] == "2310.00001"


@pytest.mark.parametrize("text", ["", "the of", '"""', "NOT AND OR", "çikolatalı kek tarifi"])
def test_odd_inputs_do_not_crash(small_db, text):
    conn, _ = small_db
    assert isinstance(search(conn, text), list)


def test_gate_never_uses_unknown_words(small_db):
    from core.search_bm25 import build_term_df, gated_query
    conn, _ = small_db
    assert build_term_df(conn) > 0
    q = gated_query(conn, "zzqx robot grasping policies reinforcement", gate=2)
    gate_part = q.split(" AND ")[0]
    assert "zzqx" not in gate_part and "zzqx" in q


def test_gate_falls_back_when_few_known_words(small_db):
    from core.search_bm25 import build_term_df, gated_query
    conn, _ = small_db
    build_term_df(conn)
    assert " AND " not in gated_query(conn, "robot kavrama öğrenmesi", gate=2)


def test_gated_search_finds_same_paper(small_db):
    from core.search_bm25 import build_term_df
    conn, _ = small_db
    build_term_df(conn)
    text = "reinforcement learning robot grasping policies"
    assert search(conn, text, k=1, gate=2)[0][0] == search(conn, text, k=1)[0][0] == "2310.00002"
