import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

from core import config, embeddings
from core.db import connect
from core.features import QUERY_FEATURES, RANK_FEATURES, FeatureBuilder
from core.search_bm25 import has_term_df
from core.search_dense import DenseIndex


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--vectors-dir", type=Path, required=True)
    p.add_argument("--field-model", type=Path, default=None)
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--add-titles", action="store_true")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args()

    conn = connect(args.db)
    if not has_term_df(conn):
        raise SystemExit("term_df tablosu yok. Önce: python -m scripts.build_term_df")
    with open(args.queries, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.add_titles:
        targets = sorted({r["target_id"] for r in rows if r.get("target_id")})
        titles = dict(conn.execute(f"SELECT id, title FROM papers WHERE id IN ({','.join('?' * len(targets))})", targets))
        rows += [{"qid": f"train-title-{pid}", "type": "title", "lang": "en", "query": titles[pid], "target_id": pid}
                 for pid in targets if pid in titles]

    index = DenseIndex.load(args.vectors_dir)
    encoder = embeddings.Encoder(index.meta["model"], max_seq_length=index.meta["max_seq_length"])
    field_model = joblib.load(args.field_model) if args.field_model else None
    builder = FeatureBuilder(conn, index, encoder, field_model)

    cand_rows, query_rows = [], []
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        for row, res in zip(batch, builder.build([r["query"] for r in batch])):
            target = row.get("target_id")
            for pk, pid, x in zip(res["pks"], res["ids"], res["X"]):
                cand_rows.append([row["qid"], pk, pid, int(pid == target), *x.tolist()])
            query_rows.append([row["qid"], row["type"], row["lang"], target, int(target in res["ids"]),
                               len(res["pks"]), *res["query"].tolist()])
        print(f"  {min(start + args.batch, len(rows)):,} / {len(rows):,} sorgu", end="\r", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cands = pd.DataFrame(cand_rows, columns=["qid", "pk", "id", "label", *RANK_FEATURES])
    queries = pd.DataFrame(query_rows, columns=["qid", "type", "lang", "target_id", "target_in_candidates",
                                                "n_candidates", *QUERY_FEATURES])
    cands.to_parquet(args.out_dir / f"{args.split}_candidates.parquet", index=False)
    queries.to_parquet(args.out_dir / f"{args.split}_queries.parquet", index=False)

    print(f"\n{args.split}: {len(queries):,} sorgu, ortalama {queries.n_candidates.mean():.0f} aday")
    with_target = queries[queries.target_id.notna()]
    recall = defaultdict(float)
    for key, g in with_target.groupby(["type", "lang"]):
        recall[f"{key[0]}/{key[1]}"] = round(g.target_in_candidates.mean(), 4)
    if recall:
        print("Hedef makalenin aday havuzunda olma oranı:", dict(recall))


if __name__ == "__main__":
    main()
