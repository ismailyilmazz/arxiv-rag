import json

import pytest

from scripts.deploy import fetch_data


def fake_hub(files: dict[str, bytes], manifest_sizes: dict[str, int] | None = None):
    calls = []

    def download(repo_id, target, token):
        calls.append((repo_id, token))
        for rel, data in files.items():
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        sizes = manifest_sizes or {rel: len(data) for rel, data in files.items()}
        (target / "manifest.json").write_text(json.dumps({"files": sizes, "papers": 3}), encoding="utf-8")

    return download, calls


FILES = {"papers.db": b"db", "vectors/vectors.npy": b"vvvv", "vectors/pks.npy": b"pp", "models/ranker.txt": b"r"}


def test_fetch_downloads_and_verifies(tmp_path):
    download, calls = fake_hub(FILES)
    manifest = fetch_data.fetch(tmp_path / "data", "user/repo", "tok", downloader=download)
    assert calls == [("user/repo", "tok")] and manifest["papers"] == 3
    assert fetch_data.verify(tmp_path / "data") == []


def test_missing_or_truncated_files_are_reported(tmp_path):
    sizes = {**{k: len(v) for k, v in FILES.items()}, "models/guard.joblib": 10, "papers.db": 99}
    download, _ = fake_hub(FILES, sizes)
    with pytest.raises(SystemExit) as error:
        fetch_data.fetch(tmp_path / "data", "user/repo", downloader=download)
    message = str(error.value)
    assert "eksik: models/guard.joblib" in message and "boyutu farklı: papers.db" in message


def test_verify_without_manifest(tmp_path):
    assert fetch_data.verify(tmp_path) == ["manifest.json yok"]
