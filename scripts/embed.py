import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import config, embeddings
from core.db import connect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(embeddings.MODELS), required=True)
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pks", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--chunk", type=int, default=20_000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    args = p.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard, 0 ile --num-shards arasında olmalı.")

    conn = connect(args.db)
    if args.pks:
        pks = np.load(args.pks).astype(np.int64)
    else:
        pks = np.array([r[0] for r in conn.execute("SELECT pk FROM papers ORDER BY pk")], dtype=np.int64)
    pks = pks[args.shard::args.num_shards]

    encoder = embeddings.Encoder(args.model, max_seq_length=args.max_seq_length)
    dim = encoder.encode_passages(["probe"]).shape[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.lib.format.open_memmap(args.out_dir / "vectors.npy", mode="w+", dtype=np.float16, shape=(len(pks), dim))
    print(f"{args.model}: {len(pks):,} makale, {dim} boyut, parça {args.shard + 1}/{args.num_shards} -> {args.out_dir}")

    start = time.perf_counter()
    for i in range(0, len(pks), args.chunk):
        chunk = pks[i:i + args.chunk].tolist()
        rows = conn.execute(
            f"SELECT pk, title, abstract FROM papers WHERE pk IN ({','.join('?' * len(chunk))})", chunk)
        text_of = {r[0]: embeddings.passage_text(r[1], r[2]) for r in rows}
        vectors[i:i + len(chunk)] = encoder.encode_passages([text_of[pk] for pk in chunk], args.batch_size)
        done = i + len(chunk)
        rate = done / (time.perf_counter() - start)
        print(f"  {done:,} / {len(pks):,}  ({rate:,.0f} makale/sn)", flush=True)

    vectors.flush()
    np.save(args.out_dir / "pks.npy", pks)
    meta = {
        "model": args.model,
        "hf_id": encoder.spec.hf_id,
        "dim": int(dim),
        "count": int(len(pks)),
        "max_seq_length": args.max_seq_length,
        "seconds": round(time.perf_counter() - start, 1),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Bitti: {meta['seconds']} sn")


if __name__ == "__main__":
    main()
