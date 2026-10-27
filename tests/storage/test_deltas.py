import json

import numpy as np

from arxiv_rag.storage import deltas
from arxiv_rag.storage.db import COLUMNS, connect, init_db, upsert_papers
from tests.conftest import make_row


def row(pid, title):
    values = {c: "" for c in COLUMNS}
    values.update(id=pid, title=title, abstract=f"Abstract of {pid}")
    return values


def base(tmp_path):
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(conn, [make_row(f"2601.0000{i}", f"Old {i}", "abstract") for i in range(1, 4)])
    pks = np.array([r[0] for r in conn.execute("SELECT pk FROM papers ORDER BY pk")], dtype=np.int64)
    conn.close()
    (tmp_path / "vectors").mkdir()
    np.save(tmp_path / "vectors" / "vectors.npy", np.zeros((3, 4), dtype=np.float16))
    np.save(tmp_path / "vectors" / "pks.npy", pks)
    return tmp_path


def delta(tmp_path, date, rows, value):
    rel = f"deltas/{date}"
    files = deltas.write_delta(rows, np.full((len(rows), 4), value, dtype=np.float32), tmp_path / rel)
    return {"date": date, "path": rel, "files": files, "records": len(rows)}


def titles(tmp_path):
    conn = connect(tmp_path / "papers.db")
    out = dict(conn.execute("SELECT id, title FROM papers").fetchall())
    conn.close()
    return out


def test_new_papers_are_added_and_updated_ones_replace_the_old_version(tmp_path):
    data = base(tmp_path)
    entry = delta(data, "2026-10-27", [row("2601.00002", "New title"), row("2610.00009", "Brand new")], 1.0)
    summary = deltas.apply_deltas(data, {"deltas": [entry]})
    assert summary == {"applied": ["2026-10-27"], "errors": [], "added": 1, "updated": 1}
    assert titles(data)["2601.00002"] == "New title" and "2610.00009" in titles(data)
    vectors, pks = np.load(data / "vectors" / "vectors.npy"), np.load(data / "vectors" / "pks.npy")
    assert vectors.shape == (4, 4) and len(pks) == 4 and vectors[1].tolist() == [1, 1, 1, 1]
    assert json.loads((data / deltas.STATE_FILE).read_text())["added"] == 1


def test_a_broken_delta_is_skipped_without_touching_the_database_and_later_ones_still_apply(tmp_path):
    data = base(tmp_path)
    broken = delta(data, "2026-10-26", [row("2610.00001", "Broken")], 2.0)
    broken["files"]["vectors.npy"] += 1
    good = delta(data, "2026-10-27", [row("2610.00002", "Good")], 3.0)
    summary = deltas.apply_deltas(data, {"deltas": [good, broken]})
    assert summary["applied"] == ["2026-10-27"] and summary["errors"][0].startswith("2026-10-26")
    assert "2610.00001" not in titles(data) and "2610.00002" in titles(data)


def test_the_same_new_paper_in_two_deltas_gets_one_vector(tmp_path):
    data = base(tmp_path)
    first = delta(data, "2026-10-26", [row("2610.00005", "v1")], 4.0)
    second = delta(data, "2026-10-27", [row("2610.00005", "v2")], 5.0)
    summary = deltas.apply_deltas(data, {"deltas": [first, second]})
    vectors = np.load(data / "vectors" / "vectors.npy")
    assert (summary["added"], summary["updated"]) == (1, 1) and vectors.shape[0] == 4 and vectors[3][0] == 5
    assert titles(data)["2610.00005"] == "v2"


def test_without_deltas_nothing_changes_and_same_date_entries_are_replaced(tmp_path):
    data = base(tmp_path)
    assert deltas.apply_deltas(data, {}) == {"applied": [], "errors": [], "added": 0, "updated": 0}
    manifest = deltas.add_delta(
        {"deltas": [{"date": "2026-10-27", "records": 1}]}, {"date": "2026-10-27", "records": 2}
    )
    assert manifest["deltas"] == [{"date": "2026-10-27", "records": 2}]
