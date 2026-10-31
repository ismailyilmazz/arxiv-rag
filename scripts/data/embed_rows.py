import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Bir parquet dosyasındaki kayıtları ana veriyle aynı ayarlarla gömer.")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="granite-311m-384")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-seq-length", type=int, default=512)
    args = p.parse_args()
    import pandas as pd

    from arxiv_rag.search.embeddings import Encoder
    from scripts.data.update_data import embed

    frame = pd.read_parquet(args.rows)
    rows = frame[["id", "title", "abstract"]].to_dict("records")
    encoder = Encoder(args.model, max_seq_length=args.max_seq_length)
    vectors = embed(rows, args.model, encoder=encoder, batch_size=args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "vectors.npy", vectors.astype(np.float16))
    np.save(args.out / "ids.npy", np.array([r["id"] for r in rows], dtype=str))
    print(f"{len(rows)} kayıt gömüldü: {args.out}")


if __name__ == "__main__":
    main()
