import numpy as np

from arxiv_rag.search.dense import DenseIndex
from arxiv_rag.search.related import related
from arxiv_rag.storage.db import connect, init_db, upsert_papers
from tests.conftest import make_row


def _db(tmp_path, ids):
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(
        conn, [make_row(pid, f"Title {pid}", f"First sentence of {pid}. Second one. Third one.") for pid in ids]
    )
    return conn


def test_related_finds_nearest_and_excludes_seeds(tmp_path):
    conn = _db(tmp_path, ["2101.00001", "2101.00002", "2101.00003", "2101.00004"])
    pk = dict(conn.execute("SELECT id, pk FROM papers").fetchall())
    vectors = np.array([[1, 0, 0], [0.9, 0.1, 0], [0.5, 0.5, 0], [0, 0, 1]], dtype=np.float16)
    pks = np.array([pk["2101.00001"], pk["2101.00002"], pk["2101.00003"], pk["2101.00004"]])
    index = DenseIndex(vectors, pks, {"model": "x"})
    found = related(conn, index, ["2101.00001"], k=2)
    assert [r["id"] for r in found] == ["2101.00002", "2101.00003"]
    assert related(conn, index, ["9999.99999"], k=2) == []
