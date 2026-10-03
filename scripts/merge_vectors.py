import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("shards", type=Path, nargs="+")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    metas = [json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in args.shards]
    for key in ("model", "hf_id", "dim", "max_seq_length"):
        values = {m[key] for m in metas}
        if len(values) != 1:
            raise SystemExit(f"Parçalar uyuşmuyor, {key}: {values}")

    pks_list = [np.load(d / "pks.npy") for d in args.shards]
    vector_list = [np.load(d / "vectors.npy", mmap_mode="r") for d in args.shards]
    for d, pks, vectors in zip(args.shards, pks_list, vector_list):
        if len(pks) != len(vectors):
            raise SystemExit(f"{d}: vektör ve kimlik sayısı farklı.")
    pks = np.concatenate(pks_list)
    if len(np.unique(pks)) != len(pks):
        raise SystemExit("Aynı makale birden fazla parçada var.")

    dim = metas[0]["dim"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(args.out_dir / "vectors.npy", mode="w+", dtype=np.float16, shape=(len(pks), dim))
    start = 0
    for vectors in vector_list:
        out[start:start + len(vectors)] = vectors
        start += len(vectors)
    out.flush()
    np.save(args.out_dir / "pks.npy", pks)

    meta = {
        **metas[0],
        "count": int(len(pks)),
        "shards": len(metas),
        "seconds": max(m["seconds"] for m in metas),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Birleştirildi: {len(pks):,} makale, {len(metas)} parça -> {args.out_dir}")


if __name__ == "__main__":
    main()
