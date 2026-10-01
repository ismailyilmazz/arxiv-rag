import pytest

from core.db import COLUMNS, connect, init_db, upsert_papers


def make_row(pid, title, abstract):
    values = {c: None for c in COLUMNS}
    values.update(id=pid, title=title, abstract=abstract)
    return tuple(values[c] for c in COLUMNS)


@pytest.fixture
def small_db(tmp_path):
    path = tmp_path / "papers.db"
    conn = connect(path)
    init_db(conn)
    upsert_papers(conn, [
        make_row("2310.00001", "Graph neural networks for traffic forecasting",
                 "We forecast road traffic with graph neural networks."),
        make_row("2310.00002", "Reinforcement learning for robot grasping",
                 "Robots learn grasping policies with reinforcement learning."),
        make_row("hep-th/9901001", "Black hole entropy in string theory",
                 "We count microstates of black holes."),
    ])
    yield conn, path
    conn.close()
