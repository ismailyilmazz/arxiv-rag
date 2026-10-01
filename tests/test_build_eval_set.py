import json
import sys

from core import llm
from core.splits import split_of
from scripts import build_eval_set, evaluate
from tests.conftest import make_row
from core.db import connect, init_db, upsert_papers


def _fake_llm(prompt, model, **kwargs):
    if "everyday questions" in prompt:
        return {"questions": ["best pizza near me", "how to fix a leaking tap", "best pizza near me"]}
    return {"en": "robot grasp learning", "tr": "robot kavrama öğrenmesi"}


def test_build_and_evaluate_end_to_end(tmp_path, monkeypatch):
    # Test havuzuna düşen kimlikleri bul ve küçük bir veritabanı kur
    test_ids = [f"2310.{i:05d}" for i in range(3000) if split_of(f"2310.{i:05d}") == "test"][:5]
    db = tmp_path / "papers.db"
    conn = connect(db)
    init_db(conn)
    upsert_papers(conn, [make_row(pid, f"Robot grasping study {pid}", "Robots grasp objects.")
                         for pid in test_ids])
    upsert_papers(conn, [make_row("2310.99999", "Unrelated train paper", "Nothing here.")])
    conn.close()

    out = tmp_path / "queries.jsonl"
    monkeypatch.setattr(llm, "chat_json", _fake_llm)
    monkeypatch.setattr(llm, "_client", lambda: None)
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--out", str(out),
                                      "--n-title", "3", "--n-synthetic", "2", "--n-offtopic", "5"])
    build_eval_set.main()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    types = [(r["type"], r["lang"]) for r in rows]
    assert types.count(("title", "en")) == 3
    assert types.count(("synthetic", "en")) == 2 and types.count(("synthetic", "tr")) == 2
    assert types.count(("offtopic", "en")) == 2          # tekrar eden soru atıldı
    assert all(split_of(r["target_id"]) == "test" for r in rows if r["target_id"])

    # İkinci çalıştırma kaldığı yerden devam eder, hiçbir şeyi tekrar eklemez
    build_eval_set.main()
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(rows)

    # Aynı set ile değerlendirme uçtan uca çalışmalı
    monkeypatch.setattr(evaluate.config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--queries", str(out)])
    evaluate.main()
    report = json.loads((tmp_path / "results" / "bm25_6.json").read_text(encoding="utf-8"))
    assert report["groups"]["title/en"]["n"] == 3
    assert report["groups"]["title/en"]["hit@10"] == 1.0


def test_title_overlap():
    assert build_eval_set.title_overlap("graph neural networks", "Graph Neural Networks for X") == 1.0
    assert build_eval_set.title_overlap("traffic prediction", "Graph Neural Networks for X") == 0.0


def test_redo_synthetic_keeps_other_rows(tmp_path, monkeypatch):
    test_ids = [f"2310.{i:05d}" for i in range(3000) if split_of(f"2310.{i:05d}") == "test"][:3]
    db = tmp_path / "papers.db"
    conn = connect(db)
    init_db(conn)
    upsert_papers(conn, [make_row(pid, f"Robot grasping study {pid}", "Robots grasp objects.")
                         for pid in test_ids])
    conn.close()
    out = tmp_path / "queries.jsonl"
    monkeypatch.setattr(llm, "_client", lambda: None)
    base = ["x", "--db", str(db), "--out", str(out), "--n-title", "3", "--n-synthetic", "2", "--n-offtopic", "5"]

    monkeypatch.setattr(llm, "chat_json", _fake_llm)
    monkeypatch.setattr(sys, "argv", base)
    build_eval_set.main()

    def better_llm(prompt, model, **kwargs):
        if "everyday questions" in prompt:
            raise AssertionError("konu dışı sorgular yeniden üretilmemeli")
        return {"en": "learning to pick up objects", "tr": "nesne kavramayı öğrenme"}

    monkeypatch.setattr(llm, "chat_json", better_llm)
    monkeypatch.setattr(sys, "argv", base + ["--redo-synthetic"])
    build_eval_set.main()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    synth = [r["query"] for r in rows if r["type"] == "synthetic"]
    assert synth.count("learning to pick up objects") == 2 and len(synth) == 4
    assert sum(r["type"] == "title" for r in rows) == 3
    assert sum(r["type"] == "offtopic" for r in rows) == 4   # dil başına 2, ikisi de korundu