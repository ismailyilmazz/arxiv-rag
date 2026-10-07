import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from core import config, search_bm25
from core.db import connect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path)
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--out-dir", type=Path, default=config.DATA_DIR)
    args = p.parse_args()

    tmp = args.out_dir / "_bundle"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(args.bundle) as z:
        z.extractall(tmp)

    conn = connect(args.db)
    pk_of = dict(conn.execute("SELECT id, pk FROM papers").fetchall())
    ids = np.load(tmp / "ids.npy")
    vectors = np.load(tmp / "vectors.npy")
    keep = [i for i, pid in enumerate(ids) if str(pid) in pk_of]
    vec_dir = args.out_dir / "vectors_dev"
    vec_dir.mkdir(parents=True, exist_ok=True)
    np.save(vec_dir / "vectors.npy", vectors[keep])
    np.save(vec_dir / "pks.npy", np.array([pk_of[str(ids[i])] for i in keep], dtype=np.int64))
    meta = json.loads((tmp / "meta.json").read_text(encoding="utf-8"))
    meta["count"] = len(keep)
    (vec_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    models_dir = args.out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for f in (tmp / "models").iterdir():
        shutil.copy(f, models_dir / f.name)
    if not search_bm25.has_term_df(conn):
        search_bm25.build_term_df(conn)
    shutil.rmtree(tmp)
    print(f"Vektörler: {len(keep):,} / {len(ids):,} makale yerel veritabanıyla eşleşti -> {vec_dir}")
    print(f"Modeller: {sorted(f.name for f in models_dir.iterdir())} -> {models_dir}")
    print("term_df hazır.")


if __name__ == "__main__":
    main()
