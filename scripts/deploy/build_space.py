import argparse
import shutil
import tomllib
from pathlib import Path

from arxiv_rag import config

EXCLUDED = {"torch"}


def requirements(pyproject: Path) -> list[str]:
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    wanted = project["dependencies"] + project["optional-dependencies"]["embed"]
    return [r for r in wanted if r.split("=")[0].split("<")[0].split(">")[0].strip() not in EXCLUDED]


def build(out: Path, root: Path = config.ROOT) -> Path:
    dist = root / "frontend" / "dist"
    if not (dist / "index.html").exists():
        raise SystemExit("Derlenmiş arayüz yok. Önce: cd frontend && npm ci && npm run build")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("app.py", "README.md"):
        shutil.copy(root / "deploy" / "huggingface" / name, out / name)
    (out / "requirements.txt").write_text("\n".join(requirements(root / "pyproject.toml")) + "\n", encoding="utf-8")
    shutil.copytree(root / "arxiv_rag", out / "arxiv_rag", ignore=ignore)
    (out / "scripts" / "deploy").mkdir(parents=True)
    for rel in ("scripts/__init__.py", "scripts/deploy/__init__.py", "scripts/deploy/fetch_data.py"):
        shutil.copy(root / rel, out / rel)
    shutil.copytree(dist, out / "frontend" / "dist")
    return Path(shutil.make_archive(str(out), "zip", out))


def push(out: Path, space_id: str, api=None) -> str:
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    commit = api.upload_folder(
        repo_id=space_id, repo_type="space", folder_path=str(out), commit_message="Uygulama güncellemesi"
    )
    return commit.commit_url


def main() -> None:
    p = argparse.ArgumentParser(description="Hugging Face Space paketini repodan üretir.")
    p.add_argument("--out", type=Path, default=config.ROOT / "build" / "hf-space")
    p.add_argument("--push", metavar="SPACE_ID", help="Paketi bu Space'e yükler, örneğin ismailylmz283/arxiv-rag")
    args = p.parse_args()
    archive = build(args.out)
    files = sum(1 for f in args.out.rglob("*") if f.is_file())
    print(f"{files} dosya: {args.out}")
    print(f"Zip: {archive}")
    if args.push:
        print(f"Yüklendi: {push(args.out, args.push)}")


if __name__ == "__main__":
    main()
