import argparse
import time
from pathlib import Path

from arxiv_rag import config
from arxiv_rag.search import bm25
from arxiv_rag.storage.db import connect


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    args = p.parse_args()

    conn = connect(args.db)
    start = time.perf_counter()
    count = bm25.build_term_df(conn)
    print(f"term_df hazır: {count:,} terim ({time.perf_counter() - start:.1f} sn) -> {args.db}")


if __name__ == "__main__":
    main()
