import argparse
import json
from pathlib import Path

import numpy as np

from core import config
from core.db import connect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--queries", type=Path, default=config.EVAL_QUERIES_PATH)
    p.add_argument("--size", type=int, default=200_000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    conn = connect(args.db)
    with open(args.queries, encoding="utf-8") as f:
        target_ids = sorted({json.loads(line)["target_id"] for line in f if line.strip()} - {None})
    rows = conn.execute(f"SELECT pk FROM papers WHERE id IN ({','.join('?' * len(target_ids))})", target_ids)
    targets = np.array([r[0] for r in rows], dtype=np.int64)

    all_pks = np.array([r[0] for r in conn.execute("SELECT pk FROM papers")], dtype=np.int64)
    others = np.setdiff1d(all_pks, targets)
    rng = np.random.default_rng(args.seed)
    n_others = max(0, min(args.size - len(targets), len(others)))
    subset = np.sort(np.concatenate([targets, rng.choice(others, n_others, replace=False)]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, subset)
    print(f"Alt küme: {len(subset):,} makale ({len(targets)} hedef + {n_others:,} rastgele) -> {args.out}")


if __name__ == "__main__":
    main()
