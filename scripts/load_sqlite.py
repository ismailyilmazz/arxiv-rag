"""Kaggle'da üretilen parquet dosyasını SQLite deposuna yükler.

Dosya parça parça okunur. Böylece 30 binlik örnek de, 3 milyonluk tam korpus da
aynı betikle ve belleği şişirmeden yüklenir.

Kullanım (proje kökünden):
    python -m scripts.load_sqlite data/dev_sample.parquet
"""
import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from core.arxiv_ids import abs_url
from core.config import DB_PATH
from core.db import COLUMNS, connect, init_db, upsert_papers

BATCH = 5_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    pf = pq.ParquetFile(args.parquet)
    total = pf.metadata.num_rows
    print(f"{total:,} makale okunacak: {args.parquet}")

    conn = connect(args.db)
    init_db(conn)

    start = time.perf_counter()
    done = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=list(COLUMNS)):
        rows = [tuple(r[c] for c in COLUMNS) for r in batch.to_pylist()]
        upsert_papers(conn, rows)
        done += len(rows)
        print(f"  {done:,} / {total:,}", end="\r")
    print(f"\nYükleme bitti ({time.perf_counter() - start:.1f} sn)")

    # Kontrol: toplam sayı ve linki doğru üretilen rastgele 3 makale
    count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"Depodaki toplam makale: {count:,}  ({args.db})")
    for r in conn.execute("SELECT id, title FROM papers ORDER BY RANDOM() LIMIT 3"):
        print(f"  {abs_url(r['id'])}  {r['title'][:70]}")
    conn.close()


if __name__ == "__main__":
    main()
