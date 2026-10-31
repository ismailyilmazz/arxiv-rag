import json

import numpy as np

from scripts.data import update_data
from tests.storage.test_oai import page, record

MANIFEST = {"created_at": "2026-10-18T01:37:56+00:00", "papers": 3, "files": {}, "embedding_model": "m"}


class Api:
    token = "t"

    def __init__(self, role="write"):
        self.commits, self.restarts, self.role = [], [], role

    def whoami(self):
        return {"auth": {"accessToken": {"role": self.role}}}

    def create_commit(self, **kw):
        self.commits.append(kw)

    def restart_space(self, repo_id):
        self.restarts.append(repo_id)


def reader(state=None):
    return lambda api, repo, name: MANIFEST if name == "manifest.json" else state


def go(tmp_path, api, get, idle=lambda url: (True, "boş"), **kw):
    return update_data.run(
        api,
        "me@x.org",
        read=kw.pop("read", reader({"until": "2026-10-26"})),
        encode=lambda rows, model: np.ones((len(rows), 4)),
        get=get,
        idle=idle,
        workdir=tmp_path,
        until="2026-10-27",
        **kw,
    )


def test_new_records_are_published_as_one_commit_and_the_space_restarts(tmp_path):
    api = Api()
    assert go(tmp_path, api, lambda url, c: page([record("2610.00001"), record("2610.00002")])) == 0
    paths = [op.path_in_repo for op in api.commits[0]["operations"]]
    assert paths == [
        "deltas/2026-10-27/papers.parquet",
        "deltas/2026-10-27/vectors.npy",
        "deltas/2026-10-27/ids.npy",
        "manifest.json",
        "state.json",
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["deltas"][0]["from"] == "2026-10-26" and manifest["deltas"][0]["records"] == 2
    assert json.loads((tmp_path / "state.json").read_text())["until"] == "2026-10-27" and api.restarts == [
        update_data.SPACE_ID
    ]


def test_failed_harvest_or_quality_check_publishes_nothing(tmp_path):
    def down(url, c):
        raise RuntimeError("arXiv kapalı")

    api = Api()
    assert go(tmp_path, api, down) == 1 and api.commits == []
    empty = page([record(f"2610.{i:05d}", title="") for i in range(1, 20)])
    assert go(tmp_path, api, lambda url, c: empty) == 1 and api.commits == []


def test_busy_space_is_not_restarted_and_dry_run_publishes_nothing(tmp_path):
    api = Api()
    assert go(tmp_path, api, lambda url, c: page([record("2610.00001")]), idle=lambda url: (False, "dolu")) == 0
    assert len(api.commits) == 1 and api.restarts == []
    dry = Api()
    assert go(tmp_path, dry, lambda url, c: page([record("2610.00001")]), dry_run=True) == 0 and dry.commits == []


def test_first_run_starts_from_the_data_date_and_empty_days_only_move_the_state(tmp_path):
    api, urls = Api(), []
    none = '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="noRecordsMatch"/></OAI-PMH>'
    assert go(tmp_path, api, lambda url, c: urls.append(url) or none, read=reader(None)) == 0
    assert "from=2026-10-18" in urls[0]
    assert [op.path_in_repo for op in api.commits[0]["operations"]] == [
        "manifest.json",
        "state.json",
    ] and api.restarts == []


def test_a_read_only_token_stops_before_any_harvest(tmp_path):
    calls = []
    api = Api(role="read")
    assert go(tmp_path, api, lambda url, c: calls.append(url) or page([record("2610.00001")])) == 1
    assert calls == [] and api.commits == []


def test_fine_grained_tokens_are_checked_for_repo_write():
    fine = {
        "role": "fineGrained",
        "fineGrained": {"scoped": [{"entity": {"name": "me"}, "permissions": ["repo.write"]}]},
    }

    class Fine(Api):
        def whoami(self):
            return {"auth": {"accessToken": fine}}

    assert update_data.can_write(Fine(), "me/data")[0]
    assert "doğrulanamadı" in update_data.can_write(Fine(), "other/data")[1]


def test_embedding_reports_progress_and_keeps_the_order():
    class Encoder:
        def encode_passages(self, texts, batch_size):
            return np.array([[float(t.split(".")[0][-1])] for t in texts])

    rows = [{"title": f"t{i}", "abstract": "a"} for i in range(5)]
    lines, ticks = [], iter(range(100))
    out = update_data.embed(
        rows, "m", encoder=Encoder(), chunk=2, log=lambda m, **k: lines.append(m), clock=lambda: next(ticks)
    )
    assert out[:, 0].tolist() == [0, 1, 2, 3, 4] and len(lines) == 3
    assert lines[-1].startswith("gömme: 5/5 kayıt")


def _paths(commit):
    return [(type(op).__name__, op.path_in_repo) for op in commit["operations"]]


def test_kaggle_embeds_and_the_staging_file_is_removed_in_the_same_commit(tmp_path):
    api, used = Api(), []

    def kaggle(ids, staging, model, workdir):
        used.append(staging)
        return np.full((len(ids), 4), 2.0)

    def cpu(rows, model):
        raise AssertionError("Kaggle başarılıyken CPU çalışmamalı")

    code = update_data.run(
        api,
        "me@x.org",
        read=reader({"until": "2026-10-26"}),
        encode=cpu,
        get=lambda u, c: page([record("2610.00001")]),
        idle=lambda url: (True, "boş"),
        workdir=tmp_path,
        until="2026-10-27",
        kaggle=kaggle,
    )
    assert code == 0 and used == ["staging/2026-10-27/papers.parquet"]
    assert _paths(api.commits[0]) == [("CommitOperationAdd", "staging/2026-10-27/papers.parquet")]
    assert ("CommitOperationDelete", "staging/2026-10-27/papers.parquet") in _paths(api.commits[1])


def test_when_kaggle_fails_cpu_takes_over_and_staging_is_still_removed(tmp_path, capsys):
    api, cpu = Api(), []

    def kaggle(ids, staging, model, workdir):
        raise RuntimeError("kota doldu")

    code = update_data.run(
        api,
        "me@x.org",
        read=reader({"until": "2026-10-26"}),
        encode=lambda rows, m: cpu.append(1) or np.ones((len(rows), 4)),
        get=lambda u, c: page([record("2610.00001")]),
        idle=lambda url: (True, "boş"),
        workdir=tmp_path,
        until="2026-10-27",
        kaggle=kaggle,
    )
    assert code == 0 and cpu == [1] and "kota doldu" in capsys.readouterr().out
    assert ("CommitOperationDelete", "staging/2026-10-27/papers.parquet") in _paths(api.commits[-1])


def test_dry_run_with_kaggle_removes_the_staging_file_and_publishes_nothing_else(tmp_path):
    api = Api()
    code = update_data.run(
        api,
        "me@x.org",
        read=reader({"until": "2026-10-26"}),
        encode=lambda rows, m: np.ones((len(rows), 4)),
        get=lambda u, c: page([record("2610.00001")]),
        idle=lambda url: (True, "boş"),
        workdir=tmp_path,
        until="2026-10-27",
        dry_run=True,
        kaggle=lambda ids, s, m, w: np.ones((len(ids), 4)),
    )
    assert code == 0 and [_paths(c) for c in api.commits] == [
        [("CommitOperationAdd", "staging/2026-10-27/papers.parquet")],
        [("CommitOperationDelete", "staging/2026-10-27/papers.parquet")],
    ]
