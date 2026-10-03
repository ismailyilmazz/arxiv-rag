import argparse
import time
from pathlib import Path

from core import config, search_bm25
from core.db import connect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    args = p.parse_args()

    conn = connect(args.db)
    start = time.perf_counter()
    count = search_bm25.build_term_df(conn)
    print(f"term_df hazır: {count:,} terim ({time.perf_counter() - start:.1f} sn) -> {args.db}")


if __name__ == "__main__":
    main()
