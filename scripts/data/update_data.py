import argparse
import datetime
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

from arxiv_rag.search.embeddings import passage_text
from arxiv_rag.storage.db import COLUMNS
from arxiv_rag.storage.deltas import add_delta, write_delta
from arxiv_rag.storage.oai import check, harvest, http_get
from scripts.data import kaggle_embed
from scripts.deploy import wait_live

DATA_REPO = "ismailylmz283/arxiv-rag-data"
SPACE_ID = "ismailylmz283/arxiv-rag"
SPACE_URL = "https://ismailylmz283-arxiv-rag.hf.space"
CPU_LIMIT = 30000


def read_json(api, repo: str, name: str) -> Optional[dict]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", token=api.token)
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def embed(
    rows: list[dict],
    model: str,
    encoder=None,
    chunk: int = 1000,
    log: Callable = print,
    clock=time.monotonic,
    batch_size: int = 32,
):
    import numpy as np

    if encoder is None:
        from arxiv_rag.search.embeddings import Encoder

        encoder = Encoder(model, device="cpu")
    texts = [passage_text(r["title"], r["abstract"]) for r in rows]
    parts, started = [], clock()
    for i in range(0, len(texts), chunk):
        parts.append(encoder.encode_passages(texts[i : i + chunk], batch_size=batch_size))
        done = min(i + chunk, len(texts))
        rate = done / max(clock() - started, 1e-9)
        left = (len(texts) - done) / rate if rate else 0
        log(
            f"gömme: {done}/{len(texts)} kayıt, saniyede {rate:.1f} kayıt, kalan yaklaşık {left / 60:.0f} dk",
            flush=True,
        )
    return np.concatenate(parts) if parts else np.zeros((0, 0), dtype=np.float32)


def can_write(api, repo: str) -> tuple[bool, str]:
    try:
        token = (api.whoami().get("auth") or {}).get("accessToken") or {}
    except Exception as error:
        return False, f"HF_TOKEN doğrulanamadı: {error}"
    role = token.get("role")
    if role == "write":
        return True, "HF_TOKEN yazma yetkili"
    if role == "fineGrained":
        owner = repo.split("/")[0]
        for scope in (token.get("fineGrained") or {}).get("scoped") or []:
            entity = scope.get("entity") or {}
            if entity.get("name") in (owner, repo) and "repo.write" in (scope.get("permissions") or []):
                return True, "HF_TOKEN ince ayarlı, veri deposuna yazma izni var"
        return True, "HF_TOKEN ince ayarlı; yazma izni doğrulanamadı, yayımlama adımında belli olacak"
    return False, f"HF_TOKEN yazma yetkili değil (rol: {role}); Hugging Face'te Write yetkili bir token gerekli"


def run(
    api,
    contact: str,
    repo: str = DATA_REPO,
    start: Optional[str] = None,
    until: Optional[str] = None,
    dry_run: bool = False,
    space: Optional[str] = SPACE_ID,
    space_url: str = SPACE_URL,
    read: Callable = read_json,
    encode: Callable = embed,
    get: Callable = http_get,
    idle: Callable = wait_live.idle,
    workdir: Optional[Path] = None,
    kaggle: Optional[Callable] = None,
) -> int:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    if not dry_run:
        ok, note = can_write(api, repo)
        print(note, flush=True)
        if not ok:
            print("hiçbir şey çekilmedi ya da yayımlanmadı")
            return 1
    manifest = read(api, repo, "manifest.json")
    if not manifest:
        print("manifest.json okunamadı; hiçbir şey yayımlanmadı")
        return 1
    state = read(api, repo, "state.json") or {}
    start = start or state.get("until") or manifest["created_at"][:10]
    until = until or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    if start > until:
        print(f"başlangıç {start} bitişten {until} sonra; yapılacak bir şey yok")
        return 0
    print(f"arXiv OAI-PMH: {start} -> {until}", flush=True)
    try:
        rows, skipped = harvest(start, until, contact, get)
    except Exception as error:
        print(f"veri çekilemedi, hiçbir şey yayımlanmadı, son başarılı tarih {state.get('until')} kaldı: {error}")
        return 1
    problems = check(rows)
    rows = [r for r in rows if r["title"] and r["abstract"]]
    print(f"{len(rows)} kayıt, {skipped} silinmiş ya da geçersiz kayıt atlandı", flush=True)
    if problems:
        print("kalite kontrolü geçmedi, hiçbir şey yayımlanmadı: " + "; ".join(problems))
        return 1
    new_state = {
        "until": until,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    operations, workdir = [], Path(workdir or tempfile.mkdtemp())
    staging = f"staging/{until}/papers.parquet"
    staged = False
    if rows:
        model = manifest.get("embedding_model", "granite-311m-384")
        vectors = None
        if kaggle is not None:
            try:
                import pandas as pd

                (workdir / "staging").mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows, columns=list(COLUMNS)).to_parquet(
                    workdir / "staging" / "papers.parquet", index=False
                )
                api.create_commit(
                    repo_id=repo,
                    repo_type="dataset",
                    operations=[CommitOperationAdd(staging, str(workdir / "staging" / "papers.parquet"))],
                    commit_message=f"Geçici kayıtlar {until} (Kaggle gömmesi için)",
                )
                staged = True
                vectors = kaggle([r["id"] for r in rows], staging, model, workdir)
                print(f"Kaggle GPU ile {len(vectors)} kayıt gömüldü", flush=True)
            except Exception as error:
                print(f"Kaggle ile gömülemedi ({error})", flush=True)
                vectors = None
        if vectors is None and len(rows) > CPU_LIMIT:
            print(
                f"{len(rows)} kayıt CPU ile süre sınırı içinde gömülemez (sınır {CPU_LIMIT}); hiçbir şey yayımlanmadı, "
                f"son başarılı tarih {state.get('until')} kaldı",
                flush=True,
            )
            if staged:
                api.create_commit(
                    repo_id=repo,
                    repo_type="dataset",
                    operations=[CommitOperationDelete(staging)],
                    commit_message=f"Geçici kayıtlar silindi {until}",
                )
            return 1
        if vectors is None:
            print("CPU ile gömülüyor", flush=True)
            vectors = encode(rows, model)
        if len(vectors) != len(rows):
            print("gömme sayısı kayıt sayısını tutmuyor, hiçbir şey yayımlanmadı")
            return 1
        rel = f"deltas/{until}"
        files = write_delta(rows, vectors, workdir / rel)
        entry = {"date": until, "from": start, "until": until, "records": len(rows), "path": rel, "files": files}
        manifest = add_delta(manifest, entry)
        operations += [CommitOperationAdd(f"{rel}/{name}", str(workdir / rel / name)) for name in files]
    for name, content in (("manifest.json", manifest), ("state.json", new_state)):
        (workdir / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        operations.append(CommitOperationAdd(name, str(workdir / name)))
    if dry_run:
        print(f"deneme modu: {len(operations)} dosya yayımlanacaktı, yayımlanmadı")
        if staged:
            api.create_commit(
                repo_id=repo,
                repo_type="dataset",
                operations=[CommitOperationDelete(staging)],
                commit_message=f"Geçici kayıtlar silindi {until}",
            )
        return 0
    if staged:
        operations.append(CommitOperationDelete(staging))
    api.create_commit(
        repo_id=repo, repo_type="dataset", operations=operations, commit_message=f"Haftalık güncelleme {until}"
    )
    print(f"yayımlandı: {len(rows)} kayıt, son başarılı tarih {until}", flush=True)
    if rows and space:
        ok, note = idle(space_url)
        if ok:
            api.restart_space(repo_id=space)
            print(f"Space yeniden başlatıldı ({note})")
        else:
            print(f"Space meşgul ({note}); yeni veri bir sonraki açılışta uygulanacak")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(
        description="arXiv'den yeni kayıtları çeker, gömer ve veri deposuna delta olarak yayımlar."
    )
    p.add_argument("--from", dest="start", default=None, help="Başlangıç tarihi (YYYY-AA-GG); verilmezse state.json")
    p.add_argument("--until", default=None)
    p.add_argument("--repo", default=os.getenv("HF_DATA_REPO", DATA_REPO))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-restart", action="store_true")
    args = p.parse_args()
    contact = os.getenv("ARXIV_CONTACT_EMAIL", "")
    if not contact or not os.getenv("HF_TOKEN"):
        print("ARXIV_CONTACT_EMAIL ve HF_TOKEN gerekli")
        sys.exit(1)
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    space = None if args.no_restart else SPACE_ID
    kaggle = None
    if kaggle_embed.ready():
        sha = os.getenv("GITHUB_SHA") or "main"
        packages = kaggle_embed.packages(Path(__file__).resolve().parents[2] / "pyproject.toml")

        def kaggle(ids, staging, model, workdir):
            kid = kaggle_embed.kernel_id()
            files = kaggle_embed.kernel_files(kid, sha, args.repo, staging, model, packages)
            return kaggle_embed.embed_on_kaggle(ids, kid, files, workdir)

    else:
        print("Kaggle anahtarları yok; gömme CPU ile yapılacak", flush=True)
    sys.exit(run(api, contact, args.repo, args.start, args.until, args.dry_run, space, kaggle=kaggle))


if __name__ == "__main__":
    main()
