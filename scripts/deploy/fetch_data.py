import argparse
import json
import os
from pathlib import Path
from typing import Callable, Optional

from arxiv_rag import config

DEFAULT_REPO = "ismailylmz283/arxiv-rag-data"
PATTERNS = ["papers.db", "vectors/*", "models/*", "manifest.json"]


def hub_download(repo_id: str, target: Path, token: Optional[str]) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=target, allow_patterns=PATTERNS, token=token)


def verify(target: Path) -> list[str]:
    manifest_path = target / "manifest.json"
    if not manifest_path.exists():
        return ["manifest.json yok"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for rel, size in manifest["files"].items():
        path = target / rel
        if not path.exists():
            problems.append(f"eksik: {rel}")
        elif path.stat().st_size != size:
            problems.append(f"boyutu farklı: {rel} ({path.stat().st_size} / {size} bayt)")
    return problems


def fetch(target: Path, repo_id: str, token: Optional[str] = None, downloader: Optional[Callable] = None) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    (downloader or hub_download)(repo_id, target, token)
    problems = verify(target)
    if problems:
        raise SystemExit("Veri eksik ya da bozuk:\n" + "\n".join(problems))
    return json.loads((target / "manifest.json").read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description="Sunucu verisini Hugging Face veri deposundan indirir ve doğrular.")
    p.add_argument("--target", type=Path, default=config.DATA_DIR)
    p.add_argument("--repo", default=os.getenv("HF_DATA_REPO", DEFAULT_REPO))
    p.add_argument("--check", action="store_true", help="İndirmeden sadece doğrula.")
    args = p.parse_args()
    if args.check:
        problems = verify(args.target)
        print("\n".join(problems) if problems else "Veri tam.")
        raise SystemExit(1 if problems else 0)
    manifest = fetch(args.target, args.repo, os.getenv("HF_TOKEN") or None)
    total = sum(manifest["files"].values()) / 1e9
    print(f"{len(manifest['files'])} dosya, {total:.2f} GB, {args.target} altında hazır.")
    for key in ("papers", "embedding_model", "scikit_learn", "lightgbm", "created_at"):
        if key in manifest:
            print(f"  {key}: {manifest[key]}")


if __name__ == "__main__":
    main()
