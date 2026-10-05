import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

from core import config
from core.db import connect
from core.fields import field_of
from core.search_dense import DenseIndex


def texts_for(conn, pks) -> list[str]:
    out = {}
    pks = [int(p) for p in pks]
    for i in range(0, len(pks), 20_000):
        chunk = pks[i:i + 20_000]
        for pk, title, abstract in conn.execute(
                f"SELECT pk, title, abstract FROM papers WHERE pk IN ({','.join('?' * len(chunk))})", chunk):
            out[pk] = f"{title}. {abstract}"
    return [out[p] for p in pks]


def query_accuracy(predict, rows, labels) -> dict:
    result = {}
    for lang in ("en", "tr"):
        idx = [i for i, r in enumerate(rows) if r["lang"] == lang]
        if idx:
            pred = predict([rows[i]["query"] for i in idx])
            result[lang] = round(accuracy_score([labels[i] for i in idx], pred), 4)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--vectors-dir", type=Path, required=True)
    p.add_argument("--queries", type=Path, default=config.EVAL_QUERIES_PATH)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-size", type=int, default=400_000)
    p.add_argument("--test-size", type=int, default=50_000)
    p.add_argument("--test-year", type=int, default=2026)
    p.add_argument("--skip-tfidf", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    conn = connect(args.db)
    index = DenseIndex.load(args.vectors_dir)
    pos = index._positions()
    fields = np.empty(len(index), dtype=object)
    years = np.zeros(len(index), dtype=np.int32)
    for pk, category, published in conn.execute("SELECT pk, primary_category, published FROM papers"):
        if pk < len(pos) and pos[pk] >= 0:
            fields[pos[pk]] = field_of(category)
            years[pos[pk]] = int(published[:4]) if published else 0

    rng = np.random.default_rng(args.seed)
    train_pool = np.flatnonzero((years > 0) & (years < args.test_year))
    test_pool = np.flatnonzero(years == args.test_year)
    train_rows = rng.choice(train_pool, min(args.train_size, len(train_pool)), replace=False)
    test_rows = rng.choice(test_pool, min(args.test_size, len(test_pool)), replace=False)
    print(f"Eğitim: {len(train_rows):,} makale (< {args.test_year}), test: {len(test_rows):,} makale ({args.test_year})")

    start = time.perf_counter()
    model = LogisticRegression(max_iter=300)
    model.fit(index.vectors[train_rows], fields[train_rows])
    groups = list(model.classes_)
    pred = model.predict(index.vectors[test_rows])
    report = {"groups": groups, "train_size": int(len(train_rows)), "test_size": int(len(test_rows)),
              "test_year": args.test_year, "embedding_lr": {
                  "accuracy": round(accuracy_score(fields[test_rows], pred), 4),
                  "macro_f1": round(f1_score(fields[test_rows], pred, average="macro"), 4),
                  "seconds": round(time.perf_counter() - start, 1)}}

    with open(args.queries, encoding="utf-8") as f:
        rows = [r for r in (json.loads(line) for line in f if line.strip()) if r["target_id"]]
    target_ids = sorted({r["target_id"] for r in rows})
    category_of = dict(conn.execute(
        f"SELECT id, primary_category FROM papers WHERE id IN ({','.join('?' * len(target_ids))})", target_ids))
    rows = [r for r in rows if r["target_id"] in category_of]
    labels = [field_of(category_of[r["target_id"]]) for r in rows]

    from core.embeddings import Encoder
    encoder = Encoder(index.meta["model"], max_seq_length=index.meta["max_seq_length"])
    report["embedding_lr"]["queries"] = query_accuracy(
        lambda texts: model.predict(encoder.encode_queries(texts)), rows, labels)

    if not args.skip_tfidf:
        start = time.perf_counter()
        vectorizer = TfidfVectorizer(max_features=200_000, sublinear_tf=True, stop_words="english", dtype=np.float32)
        x_train = vectorizer.fit_transform(texts_for(conn, index.pks[train_rows]))
        tfidf = SGDClassifier(loss="log_loss", alpha=1e-6, max_iter=20, random_state=args.seed)
        tfidf.fit(x_train, fields[train_rows])
        tfidf_pred = tfidf.predict(vectorizer.transform(texts_for(conn, index.pks[test_rows])))
        report["tfidf_sgd"] = {
            "accuracy": round(accuracy_score(fields[test_rows], tfidf_pred), 4),
            "macro_f1": round(f1_score(fields[test_rows], tfidf_pred, average="macro"), 4),
            "seconds": round(time.perf_counter() - start, 1),
            "queries": query_accuracy(lambda texts: tfidf.predict(vectorizer.transform(texts)), rows, labels)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "groups": groups, "embedding_model": index.meta["model"]}, args.out)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "field_classifier.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'model':<14}{'doğruluk':>10}{'macro-F1':>10}{'sorgu EN':>10}{'sorgu TR':>10}")
    for name in ("embedding_lr", "tfidf_sgd"):
        if name in report:
            r = report[name]
            print(f"{name:<14}{r['accuracy']:>10.3f}{r['macro_f1']:>10.3f}"
                  f"{r['queries'].get('en', 0):>10.3f}{r['queries'].get('tr', 0):>10.3f}")
    print(f"\nModel: {args.out}")


if __name__ == "__main__":
    main()
