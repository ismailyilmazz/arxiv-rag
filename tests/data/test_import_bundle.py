import json
import sys
import zipfile

import numpy as np

from arxiv_rag.search.dense import DenseIndex
from arxiv_rag.storage.db import connect, init_db, upsert_papers
from tests.conftest import make_row


def _db(tmp_path, ids):
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(
        conn, [make_row(pid, f"Title {pid}", f"First sentence of {pid}. Second one. Third one.") for pid in ids]
    )
    return conn


def test_import_bundle_maps_ids_to_local_rows(tmp_path, monkeypatch):
    from scripts.data import import_bundle

    conn = _db(tmp_path, ["2101.00001", "2101.00002"])
    pk = dict(conn.execute("SELECT id, pk FROM papers").fetchall())
    conn.close()
    src = tmp_path / "bundle"
    (src / "models").mkdir(parents=True)
    np.save(src / "vectors.npy", np.eye(3, 4, dtype=np.float16))
    np.save(src / "ids.npy", np.array(["2101.00002", "2101.00001", "9999.99999"]))
    (src / "meta.json").write_text(json.dumps({"model": "granite-311m-384", "count": 3, "max_seq_length": 512}))
    for name in ("ranker.txt", "guard.joblib", "field_clf.joblib"):
        (src / "models" / name).write_text("x")
    bundle = tmp_path / "local_bundle.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        for f in src.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(src))
    out = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["x", str(bundle), "--db", str(tmp_path / "papers.db"), "--out-dir", str(out)])
    import_bundle.main()
    index = DenseIndex.load(out / "vectors_dev")
    assert index.pks.tolist() == [pk["2101.00002"], pk["2101.00001"]] and index.meta["count"] == 2
    assert index.vectors[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert sorted(f.name for f in (out / "models").iterdir()) == ["field_clf.joblib", "guard.joblib", "ranker.txt"]
    assert not (out / "_bundle").exists()
    from arxiv_rag.search.bm25 import has_term_df

    assert has_term_df(connect(tmp_path / "papers.db"))
