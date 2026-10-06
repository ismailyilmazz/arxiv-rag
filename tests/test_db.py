from core.db import COLUMNS, connect, init_db, upsert_papers


def _row(pid, title, abstract):
    values = {c: None for c in COLUMNS}
    values.update(id=pid, title=title, abstract=abstract)
    return tuple(values[c] for c in COLUMNS)


def _search(conn, query):
    sql = """SELECT p.id FROM papers_fts f JOIN papers p ON p.pk = f.rowid
             WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts)"""
    return [r["id"] for r in conn.execute(sql, (query,))]


def test_search_stems_plural_forms(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    upsert_papers(conn, [
        _row("2310.00001", "Graph neural networks for traffic", "We forecast traffic."),
        _row("2310.00002", "Cooking with robots", "A kitchen robot."),
    ])
    assert _search(conn, "network") == ["2310.00001"]
    assert _search(conn, "robot") == ["2310.00002"]


def test_upsert_keeps_index_in_sync(tmp_path):
    conn = connect(tmp_path / "t.db")
    init_db(conn)
    upsert_papers(conn, [_row("2310.00001", "Old title about graphs", "x")])
    upsert_papers(conn, [_row("2310.00001", "New title about transformers", "x")])
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert _search(conn, "graphs") == []
    assert _search(conn, "transformer") == ["2310.00001"]


def test_connect_accepts_plain_string_path(tmp_path):
    from core.db import connect
    conn = connect(str(tmp_path / "sub" / "papers.db"))
    assert conn.execute("SELECT 1").fetchone()[0] == 1