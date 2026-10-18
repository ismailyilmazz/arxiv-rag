import shutil
import zipfile
from pathlib import Path

import pytest

from arxiv_rag import config
from scripts.deploy import build_space

REPO = config.ROOT


def fake_repo(tmp_path: Path, with_dist: bool = True) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(REPO / "arxiv_rag", root / "arxiv_rag", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "deploy" / "huggingface", root / "deploy" / "huggingface")
    (root / "scripts" / "deploy").mkdir(parents=True)
    for rel in ("scripts/__init__.py", "scripts/deploy/__init__.py", "scripts/deploy/fetch_data.py"):
        shutil.copy(REPO / rel, root / rel)
    shutil.copy(REPO / "pyproject.toml", root / "pyproject.toml")
    if with_dist:
        (root / "frontend" / "dist" / "assets").mkdir(parents=True)
        (root / "frontend" / "dist" / "index.html").write_text("<html></html>")
        (root / "frontend" / "dist" / "assets" / "app.js").write_text("x")
    return root


def test_requirements_follow_pyproject_without_torch():
    reqs = build_space.requirements(REPO / "pyproject.toml")
    assert "scikit-learn==1.6.1" in reqs and "sentence-transformers==6.1.0" in reqs
    assert not any(r.startswith("torch") for r in reqs) and not any(r.startswith("pytest") for r in reqs)


def test_build_assembles_space(tmp_path):
    root = fake_repo(tmp_path)
    archive = build_space.build(tmp_path / "out" / "hf-space", root)
    out = tmp_path / "out" / "hf-space"
    for rel in (
        "app.py",
        "README.md",
        "requirements.txt",
        "arxiv_rag/api/app.py",
        "scripts/deploy/fetch_data.py",
        "frontend/dist/index.html",
        "frontend/dist/assets/app.js",
    ):
        assert (out / rel).exists(), rel
    assert 'python_version: "3.12"' in (out / "README.md").read_text(encoding="utf-8")
    assert not list(out.rglob("__pycache__"))
    with zipfile.ZipFile(archive) as z:
        assert "app.py" in z.namelist() and "frontend/dist/index.html" in z.namelist()


def test_build_requires_built_frontend(tmp_path):
    with pytest.raises(SystemExit, match="npm run build"):
        build_space.build(tmp_path / "out", fake_repo(tmp_path, with_dist=False))


def test_encoder_device_comes_from_config(monkeypatch):
    import sys
    import types

    seen = {}

    class FakeST:
        def __init__(self, name, device=None, truncate_dim=None):
            seen["device"] = device
            self.device = types.SimpleNamespace(type=device or "cpu")

        def half(self):
            return self

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.setattr(config, "ENCODER_DEVICE", "cpu")
    from arxiv_rag.search import embeddings

    embeddings.Encoder("granite-311m-384")
    assert seen["device"] == "cpu"
    embeddings.Encoder("granite-311m-384", device="cuda")
    assert seen["device"] == "cuda"


def test_push_uploads_the_built_folder(tmp_path):
    calls = {}

    class FakeApi:
        def upload_folder(self, **kwargs):
            calls.update(kwargs)
            return type("Commit", (), {"commit_url": "https://huggingface.co/spaces/u/s/commit/abc"})()

    url = build_space.push(tmp_path, "u/s", api=FakeApi())
    assert url.endswith("/commit/abc")
    assert calls["repo_id"] == "u/s" and calls["repo_type"] == "space" and calls["folder_path"] == str(tmp_path)


def test_space_app_keeps_port_7860_for_our_server():
    source = (REPO / "deploy" / "huggingface" / "app.py").read_text(encoding="utf-8")
    assert "ssr_mode=False" in source and "port=7860" in source
    assert "demo.launch(" in source and "server_port=7861" in source
