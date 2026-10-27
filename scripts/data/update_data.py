import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from arxiv_rag.search.embeddings import passage_text
from arxiv_rag.storage.deltas import add_delta, write_delta
from arxiv_rag.storage.oai import check, harvest, http_get
from scripts.deploy import wait_live

DATA_REPO = "ismailylmz283/arxiv-rag-data"
SPACE_ID = "ismailylmz283/arxiv-rag"
SPACE_URL = "https://ismailylmz283-arxiv-rag.hf.space"


def read_json(api, repo: str, name: str) -> Optional[dict]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        path = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", token=api.token)
    except EntryNotFoundError:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def embed(rows: list[dict], model: str):
    from arxiv_rag.search.embeddings import Encoder

    encoder = Encoder(model, device="cpu")
    return encoder.encode_passages([passage_text(r["title"], r["abstract"]) for r in rows], batch_size=32)


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
) -> int:
    from huggingface_hub import CommitOperationAdd

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
    if rows:
        vectors = encode(rows, manifest.get("embedding_model", "granite-311m-384"))
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
        return 0
    api.create_commit(
        repo_id=repo, repo_type="dataset", operations=operations, commit_message=f"Günlük güncelleme {until}"
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
    sys.exit(run(api, contact, args.repo, args.start, args.until, args.dry_run, space))


if __name__ == "__main__":
    main()
