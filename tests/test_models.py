import json
import sys

import pandas as pd

from core import embeddings
from core.db import COLUMNS, connect, init_db, upsert_papers
from core.fields import field_of
from core.search_bm25 import build_term_df
from scripts import build_features, embed, evaluate, train_field_classifier, train_guard, train_ranker
from tests.test_dense_hybrid import FakeEncoder

CS = ["graph neural network node embedding", "transformer language model attention",
      "reinforcement learning policy agent", "image segmentation convolution network"]
MATH = ["prime number elliptic curve", "algebraic topology homology group",
        "partial differential equation existence", "random walk probability measure"]


def _row(pid, title, abstract, category, year):
    values = {c: None for c in COLUMNS}
    values.update(id=pid, title=title, abstract=abstract, categories=category, primary_category=category,
                  published=f"{year}-05-01")
    return tuple(values[c] for c in COLUMNS)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_field_of_maps_old_archives():
    assert field_of("cmp-lg") == "cs" and field_of("hep-th") == "hep" and field_of("cs.CL") == "cs"


def test_full_model_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "Encoder", FakeEncoder)
    db = tmp_path / "papers.db"
    conn = connect(db)
    init_db(conn)
    papers = []
    for i in range(30):
        papers.append((f"2401.{i:05d}", f"{CS[i % 4]} study{i}", f"we study {CS[i % 4]} variant{i}", "cs.LG", 2020 + i % 7))
        papers.append((f"2402.{i:05d}", f"{MATH[i % 4]} result{i}", f"we prove {MATH[i % 4]} case{i}", "math.NT", 2020 + i % 7))
    upsert_papers(conn, [_row(*p) for p in papers])
    build_term_df(conn)
    conn.close()

    vec = tmp_path / "vec"
    monkeypatch.setattr(sys, "argv", ["x", "--model", "e5-small", "--db", str(db), "--out-dir", str(vec)])
    embed.main()

    results, models, feats = tmp_path / "results", tmp_path / "models", tmp_path / "features"
    for module in (train_field_classifier, train_ranker, train_guard):
        monkeypatch.setattr(module.config, "RESULTS_DIR", results)

    eval_file, train_file, off_file = tmp_path / "eval.jsonl", tmp_path / "train.jsonl", tmp_path / "off.jsonl"
    _write(eval_file, [{"qid": f"e{i}", "type": "title", "lang": "en", "query": papers[i][1], "target_id": papers[i][0]}
                       for i in range(6)] +
           [{"qid": f"o{i}", "type": "offtopic", "lang": "en", "query": f"best pizza recipe {i}", "target_id": None}
            for i in range(4)])
    _write(train_file, [{"qid": f"t{i}", "type": "synthetic", "lang": "en", "query": papers[i][2], "target_id": papers[i][0]}
                        for i in range(6, 40)])
    _write(off_file, [{"qid": f"n{i}", "type": "offtopic", "lang": "en", "query": f"cheap flights to city {i}",
                       "target_id": None} for i in range(30)])

    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--vectors-dir", str(vec), "--queries", str(eval_file),
                                      "--out", str(models / "field_clf.joblib"), "--train-size", "40", "--test-size", "10"])
    train_field_classifier.main()
    field_report = json.loads((results / "field_classifier.json").read_text())
    assert set(field_report["groups"]) == {"cs", "math"} and "tfidf_sgd" in field_report

    for split, path, extra in (("train", train_file, ["--add-titles"]), ("offtopic", off_file, []), ("eval", eval_file, [])):
        monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--vectors-dir", str(vec), "--field-model",
                                          str(models / "field_clf.joblib"), "--queries", str(path), "--split", split,
                                          "--out-dir", str(feats), *extra])
        build_features.main()
    train_q = pd.read_parquet(feats / "train_queries.parquet")
    assert (train_q.type == "title").sum() == 34
    cands = pd.read_parquet(feats / "eval_candidates.parquet")
    assert cands.groupby("qid").label.sum().loc["e0"] == 1

    monkeypatch.setattr(sys, "argv", ["x", "--features-dir", str(feats), "--model-out", str(models / "ranker.txt")])
    train_ranker.main()
    ranker_report = json.loads((results / "ranker.json").read_text())
    assert "title/en" in ranker_report["ranker"] and "title/en" in ranker_report["dense"]

    monkeypatch.setattr(sys, "argv", ["x", "--features-dir", str(feats), "--ranker-model", str(models / "ranker.txt"),
                                      "--model-out", str(models / "guard.joblib")])
    train_guard.main()
    guard_report = json.loads((results / "guard.json").read_text())
    assert 0.0 <= guard_report["guard"]["roc_auc"] <= 1.0
    assert (models / "guard.joblib").exists()

    from core.db import connect as open_db
    from core.pipeline import SearchPipeline

    pipe = SearchPipeline.load(open_db(db), vec, models)
    out = pipe.search(papers[0][1], k=5)
    assert len(out["results"]) == 5 and out["results"][0]["id"] == papers[0][0]
    assert isinstance(out["accepted"], bool) and 0.0 <= out["guard_prob"] <= 1.0
    assert {"encode_ms", "dense_ms", "bm25_ms", "features_ms", "ranker_ms", "guard_ms", "total_ms"} <= set(out["timings_ms"])

    monkeypatch.setattr(evaluate.config, "RESULTS_DIR", results)
    monkeypatch.setattr(sys, "argv", ["x", "--method", "pipeline", "--db", str(db), "--queries", str(eval_file),
                                      "--vectors-dir", str(vec), "--models-dir", str(models)])
    evaluate.main()
    live = json.loads((results / "pipeline-e5-small_60.json").read_text(encoding="utf-8"))
    assert live["groups"]["title/en"] == ranker_report["ranker"]["title/en"]
    assert set(live["guard_accept_rate"]) == {"title/en", "offtopic/en"}
