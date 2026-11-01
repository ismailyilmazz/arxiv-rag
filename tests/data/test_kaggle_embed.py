import ast
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data import kaggle_embed

IDS = ["2610.00001", "2610.00002"]


def files():
    return kaggle_embed.kernel_files(
        "me/arxiv-rag-weekly-embed", "abc123", "me/data", "staging/x/papers.parquet", "m", ["st==6.1.0"]
    )


def fake_run(states, output="ok", push_codes=(0,)):
    calls, codes, states = [], list(push_codes), list(states)

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[2] == "push":
            return SimpleNamespace(returncode=codes.pop(0) if codes else 0, stdout="", stderr="bad accelerator")
        if cmd[2] == "status":
            return SimpleNamespace(
                returncode=0, stdout=f'me/k has status "KernelWorkerStatus.{states.pop(0)}"', stderr=""
            )
        folder = kaggle_embed.Path(cmd[cmd.index("-p") + 1])
        if output == "ok":
            np.save(folder / "vectors.npy", np.ones((2, 4), dtype=np.float16))
            np.save(folder / "ids.npy", np.array(IDS))
        elif output == "wrong":
            np.save(folder / "vectors.npy", np.ones((1, 4), dtype=np.float16))
            np.save(folder / "ids.npy", np.array(IDS[:1]))
        (folder / "arxiv-rag-weekly-embed.log").write_text("line1\nCUDA out of memory\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run, calls


def go(tmp_path, run, timeout=600):
    ticks = iter(range(0, 100000, 30))
    return kaggle_embed.embed_on_kaggle(
        IDS,
        "me/k",
        files(),
        tmp_path,
        run=run,
        sleep=lambda s: None,
        clock=lambda: next(ticks),
        timeout=timeout,
        log=lambda *a, **k: None,
    )


def test_kernel_files_are_a_valid_gpu_script_pinned_to_the_commit():
    out = files()
    meta = json.loads(out["kernel-metadata.json"])
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["kernel_type"] == "script"
    assert meta["id"].endswith("/" + meta["title"]) and meta["code_file"] == "embed.py"
    ast.parse(out["embed.py"])
    assert (
        "abc123" in out["embed.py"] and "staging/x/papers.parquet" in out["embed.py"] and "st==6.1.0" in out["embed.py"]
    )


def test_status_text_is_read_in_old_and_new_formats():
    assert kaggle_embed.status_of('has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert kaggle_embed.status_of('has status "running"') == "running"
    assert kaggle_embed.status_of('has status "error"') == "error"


def test_vectors_come_back_from_kaggle_in_the_same_order(tmp_path):
    run, calls = fake_run(["RUNNING", "COMPLETE"])
    vectors = go(tmp_path, run)
    assert vectors.shape == (2, 4) and calls[0][-2:] == ["--accelerator", kaggle_embed.ACCELERATOR]


def test_push_retries_without_the_accelerator_flag(tmp_path):
    run, calls = fake_run(["COMPLETE"], push_codes=(1, 0))
    go(tmp_path, run)
    assert "--accelerator" in calls[0] and "--accelerator" not in calls[1]


def test_failures_raise_with_the_reason(tmp_path):
    run, _ = fake_run(["ERROR"], output="log")
    with pytest.raises(kaggle_embed.KaggleFailed, match="CUDA out of memory"):
        go(tmp_path / "a", run)
    run, _ = fake_run(["COMPLETE"], output="wrong")
    with pytest.raises(kaggle_embed.KaggleFailed, match="eşleşmiyor"):
        go(tmp_path / "b", run)
    run, _ = fake_run(["RUNNING"] * 50)
    with pytest.raises(kaggle_embed.KaggleFailed, match="bitmedi"):
        go(tmp_path / "c", run, timeout=60)


def test_embedding_packages_come_from_pyproject():
    root = kaggle_embed.Path(__file__).resolve().parents[2]
    assert kaggle_embed.packages(root / "pyproject.toml") == ["sentence-transformers==6.1.0", "transformers==5.18.0"]


def test_embed_rows_writes_vectors_and_ids_in_order(tmp_path, monkeypatch):
    import sys

    import pandas as pd

    from arxiv_rag.search import embeddings
    from scripts.data import embed_rows

    class Encoder:
        def __init__(self, name, max_seq_length=512):
            self.seen = max_seq_length

        def encode_passages(self, texts, batch_size):
            return np.array([[float(t.split(".")[0][-1]), batch_size] for t in texts])

    monkeypatch.setattr(embeddings, "Encoder", Encoder)
    pd.DataFrame([{"id": f"2610.0000{i}", "title": f"t{i}", "abstract": "a"} for i in range(3)]).to_parquet(
        tmp_path / "rows.parquet"
    )
    monkeypatch.setattr(
        sys, "argv", ["embed_rows", "--rows", str(tmp_path / "rows.parquet"), "--out", str(tmp_path / "out")]
    )
    embed_rows.main()
    vectors, ids = np.load(tmp_path / "out" / "vectors.npy"), np.load(tmp_path / "out" / "ids.npy")
    assert vectors.dtype == np.float16 and vectors[:, 0].tolist() == [0, 1, 2] and vectors[0, 1] == 128
    assert ids.tolist() == ["2610.00000", "2610.00001", "2610.00002"]


def test_failure_messages_keep_the_end_of_the_error_where_the_cause_is(tmp_path):
    trace = "Traceback (most recent call last):\n" + "  File x\n" * 60 + "ModuleNotFoundError: No module named 'six'"

    def run(cmd, **kw):
        return SimpleNamespace(returncode=1, stdout="", stderr=trace)

    with pytest.raises(kaggle_embed.KaggleFailed, match="No module named 'six'"):
        go(tmp_path, run)
