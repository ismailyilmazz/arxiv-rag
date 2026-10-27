import json

import numpy as np

from scripts.data import update_data
from tests.storage.test_oai import page, record

MANIFEST = {"created_at": "2026-10-18T01:37:56+00:00", "papers": 3, "files": {}, "embedding_model": "m"}


class Api:
    token = "t"

    def __init__(self):
        self.commits, self.restarts = [], []

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
