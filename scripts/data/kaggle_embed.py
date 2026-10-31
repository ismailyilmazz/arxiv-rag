import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

KERNEL_SLUG = "arxiv-rag-weekly-embed"
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
ACCELERATOR = "NvidiaTeslaT4"
SCRIPT = """import os
import subprocess
import sys

subprocess.run(["git", "clone", "-q", {repo!r}, "/tmp/src"], check=True)
subprocess.run(["git", "-C", "/tmp/src", "checkout", "-q", {sha!r}], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-dotenv", *{packages!r}], check=True)
from huggingface_hub import hf_hub_download

rows = hf_hub_download({data_repo!r}, {staging!r}, repo_type="dataset")
env = dict(os.environ, ENCODER_DEVICE="cuda")
command = [sys.executable, "-m", "scripts.data.embed_rows", "--rows", rows, "--out", "/kaggle/working", "--model", {model!r}]
subprocess.run(command, check=True, cwd="/tmp/src", env=env)
"""


class KaggleFailed(Exception):
    pass


def ready() -> bool:
    return bool(os.getenv("KAGGLE_USERNAME") and (os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN")))


def kernel_id() -> str:
    return f"{os.environ['KAGGLE_USERNAME']}/{KERNEL_SLUG}"


def kernel_files(kid: str, sha: str, data_repo: str, staging: str, model: str, packages: list[str]) -> dict:
    script = SCRIPT.format(repo=REPO_URL, sha=sha, packages=packages, data_repo=data_repo, staging=staging, model=model)
    meta = {
        "id": kid,
        "title": kid.split("/")[1],
        "code_file": "embed.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    return {"embed.py": script, "kernel-metadata.json": json.dumps(meta, indent=2)}


def status_of(text: str) -> str:
    lowered = text.lower()
    for state in ("complete", "error", "cancel", "running", "queued"):
        if state in lowered:
            return state
    return "unknown"


def _tail(folder: Path) -> str:
    logs = sorted(folder.rglob("*.log"))
    if not logs:
        return ""
    lines = logs[0].read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return " | ".join(lines[-5:])[-600:]


def embed_on_kaggle(
    ids: list[str],
    kid: str,
    files: dict,
    workdir: Path,
    run: Callable = subprocess.run,
    sleep: Callable = time.sleep,
    clock: Callable = time.monotonic,
    timeout: float = 5400,
    interval: float = 30,
    log: Callable = print,
) -> np.ndarray:
    push_dir, out_dir = workdir / "kaggle_kernel", workdir / "kaggle_output"
    push_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (push_dir / name).write_text(content, encoding="utf-8")
    push = ["kaggle", "kernels", "push", "-p", str(push_dir)]
    result = run(push + ["--accelerator", ACCELERATOR], capture_output=True, text=True)
    if result.returncode != 0:
        log(f"Kaggle: hızlandırıcı seçilemedi, varsayılan GPU ile deneniyor ({(result.stderr or '').strip()[:200]})")
        result = run(push, capture_output=True, text=True)
    if result.returncode != 0:
        raise KaggleFailed(f"push başarısız: {(result.stderr or result.stdout or '').strip()[:300]}")
    log(f"Kaggle: {kid} başlatıldı", flush=True)
    deadline, state = clock() + timeout, "queued"
    while True:
        sleep(interval)
        answer = run(["kaggle", "kernels", "status", kid], capture_output=True, text=True)
        new = status_of(answer.stdout or "") if answer.returncode == 0 else "unknown"
        if new != state:
            log(f"Kaggle durumu: {new}", flush=True)
            state = new
        if state in ("complete", "error", "cancel"):
            break
        if clock() >= deadline:
            raise KaggleFailed(f"{int(timeout)} sn içinde bitmedi (son durum: {state})")
    out_dir.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "kernels", "output", kid, "-p", str(out_dir)], capture_output=True, text=True)
    if state != "complete":
        raise KaggleFailed(f"çalışma {state} ile bitti; günlük: {_tail(out_dir)}")
    found = {p.name: p for p in out_dir.rglob("*.npy")}
    if "vectors.npy" not in found or "ids.npy" not in found:
        raise KaggleFailed(f"çıktıda vectors.npy ya da ids.npy yok; günlük: {_tail(out_dir)}")
    vectors = np.load(found["vectors.npy"])
    got = [str(i) for i in np.load(found["ids.npy"])]
    if got != list(ids) or len(vectors) != len(ids):
        raise KaggleFailed("Kaggle çıktısındaki kimlikler ya da vektör sayısı kayıtlarla eşleşmiyor")
    return vectors


def packages(pyproject: Path) -> list[str]:
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r"^embed = \[(.*?)\]", text, re.M)
    return re.findall(r'"([^"]+)"', match.group(1)) if match else []
