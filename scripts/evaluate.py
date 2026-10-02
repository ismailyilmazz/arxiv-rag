import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from core import config, search_bm25, search_hybrid
from core.db import connect
from core.metrics import rank_of, summarize

METHODS = ("bm25", "dense", "hybrid")


def build_search(method: str, vectors_dir, n_papers: int):
    if method == "bm25":
        return search_bm25.search, n_papers, None
    if vectors_dir is None:
        raise SystemExit(f"--method {method} için --vectors-dir gerekli.")

    from core.embeddings import Encoder
    from core.search_dense import DenseIndex

    index = DenseIndex.load(vectors_dir)
    encoder = Encoder(index.meta["model"], max_seq_length=index.meta["max_seq_length"])

    def dense(conn, text, k):
        return index.search(conn, encoder.encode_queries([text])[0], k)

    if method == "dense":
        return dense, len(index), index.meta["model"]
    if len(index) != n_papers:
        raise SystemExit("Hibrit arama için vektörler tüm korpusu kapsamalı. Alt kümede sadece dense ölçülür.")
    return (lambda conn, text, k: search_hybrid.search(conn, text, k, dense)), len(index), index.meta["model"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=METHODS, default="bm25")
    p.add_argument("--vectors-dir", type=Path, default=None)
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--queries", type=Path, default=config.EVAL_QUERIES_PATH)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()

    conn = connect(args.db)
    n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    search, searched, model = build_search(args.method, args.vectors_dir, n_papers)
    label = args.method if model is None else f"{args.method}-{model}"
    with open(args.queries, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    targets = {q["target_id"] for q in queries if q["target_id"]}
    present = {r[0] for r in conn.execute(
        f"SELECT id FROM papers WHERE id IN ({','.join('?' * len(targets))})", list(targets))}
    missing = targets - present

    groups, offtopic_scores, latencies = {}, {}, []
    for i, q in enumerate(queries, start=1):
        if i % 25 == 0 or i == len(queries):
            print(f"  {i}/{len(queries)} sorgu", end="\r", flush=True)
        if q["target_id"] in missing:
            continue
        start = time.perf_counter()
        results = search(conn, q["query"], k=args.k)
        latencies.append((time.perf_counter() - start) * 1000)
        key = f"{q['type']}/{q['lang']}"
        if q["target_id"] is None:
            offtopic_scores.setdefault(key, []).append(results[0][1] if results else 0.0)
        else:
            groups.setdefault(key, []).append(rank_of(q["target_id"], [r[0] for r in results]))

    report = {
        "method": args.method,
        "model": model,
        "corpus_size": n_papers,
        "searched_docs": searched,
        "queries_file": str(args.queries.name),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "missing_targets": len(missing),
        "median_latency_ms": round(statistics.median(latencies), 1),
        "groups": {key: summarize(ranks, args.k) for key, ranks in sorted(groups.items())},
        "offtopic_top1_score": {key: round(statistics.mean(s), 3) for key, s in sorted(offtopic_scores.items())},
    }

    print(f"\nYöntem: {label} | aranan: {searched:,} makale | "
          f"ortanca gecikme: {report['median_latency_ms']} ms")
    if missing:
        print(f"Uyarı: {len(missing)} sorgunun doğru makalesi bu veritabanında yok, atlandı.")
    print(f"\n{'grup':<16}{'n':>5}{'hit@1':>9}{'hit@' + str(args.k):>9}{'mrr@' + str(args.k):>9}")
    for key, m in report["groups"].items():
        print(f"{key:<16}{m['n']:>5}{m['hit@1']:>9.3f}{m[f'hit@{args.k}']:>9.3f}{m[f'mrr@{args.k}']:>9.3f}")
    for key, s in report["offtopic_top1_score"].items():
        print(f"{key:<16} ortalama en iyi skor: {s}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / f"{label}_{searched}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRapor: {out}")


if __name__ == "__main__":
    main()
