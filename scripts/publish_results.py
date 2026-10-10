import argparse
import os
import shutil
import subprocess
from pathlib import Path

REPO = "ismailyilmazz/arxiv-rag"


def run(args: list[str], cwd: Path, secret: str = "") -> tuple[int, str]:
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    return r.returncode, out.replace(secret, "***") if secret else out


def publish(repo_dir: Path, source_dir: Path, remote: str, secret: str = "",
            message: str = "Kaggle: geniş üretim sonuçları") -> str:
    files = sorted(p for p in source_dir.glob("*") if p.suffix in (".md", ".json"))
    if not files:
        return "Gönderilecek dosya yok."
    target = repo_dir / "eval" / "generations"
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, target / f.name)
    run(["git", "config", "user.name", "kaggle-runner"], repo_dir)
    run(["git", "config", "user.email", "kaggle-runner@users.noreply.github.com"], repo_dir)
    run(["git", "add", "eval/generations"], repo_dir)
    code, out = run(["git", "commit", "-m", message], repo_dir)
    if code != 0:
        return f"Commit yapılamadı: {out[-300:]}"
    _, branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    code, out = run(["git", "pull", "--rebase", remote, branch], repo_dir, secret)
    if code != 0:
        return f"Pull başarısız: {out[-300:]}"
    code, out = run(["git", "push", remote, f"HEAD:{branch}"], repo_dir, secret)
    if code != 0:
        return f"Push başarısız: {out[-300:]}"
    return f"{len(files)} dosya repoya gönderildi ({branch}): " + ", ".join(f.name for f in files)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-dir", type=Path, default=Path("."))
    p.add_argument("--source-dir", type=Path, default=Path("/kaggle/working/generations"))
    p.add_argument("--remote", default=None)
    args = p.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")
    remote = args.remote or (f"https://x-access-token:{token}@github.com/{REPO}.git" if token else "")
    if not remote:
        print("GITHUB_TOKEN yok, sonuçlar repoya gönderilmedi. Output'taki generations.zip dosyasını kullan.")
        return
    print(publish(args.repo_dir, args.source_dir, remote, token))


if __name__ == "__main__":
    main()
