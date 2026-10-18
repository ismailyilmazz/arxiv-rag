import json
import sys

from arxiv_rag import config
from arxiv_rag.search.lang import is_turkish
from scripts.evaluation import evaluate


class FakeTranslator:
    tokens, failures = 84, 0


class FakePipeline:
    translator = FakeTranslator()

    def search(self, text, k):
        return {
            "results": [{"id": "2310.00001", "score": 1.0}, {"id": "2310.00002", "score": 0.5}],
            "accepted": True,
            "timings_ms": {"total_ms": 1.0},
            "translated_query": "graph neural networks for traffic" if is_turkish(text) else None,
        }


def test_turkish_only_run_with_translation_report(tmp_path, small_db, monkeypatch):
    queries = tmp_path / "queries.jsonl"
    rows = [
        {
            "qid": "s-tr",
            "type": "synthetic",
            "lang": "tr",
            "query": "trafik tahmini için çizge sinir ağları",
            "target_id": "2310.00001",
        },
        {
            "qid": "s-en",
            "type": "synthetic",
            "lang": "en",
            "query": "graph networks for traffic",
            "target_id": "2310.00001",
        },
        {
            "qid": "o-tr",
            "type": "offtopic",
            "lang": "tr",
            "query": "Kahve nasıl demlenir ve servis edilir?",
            "target_id": None,
        },
    ]
    queries.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(evaluate, "build_search", lambda *a, **kw: (evaluate.PipelineSearch(FakePipeline()), 3, "fake"))
    argv = ["evaluate", "--method", "pipeline", "--db", str(tmp_path / "papers.db"), "--queries", str(queries)]
    monkeypatch.setattr(sys, "argv", argv + ["--translation", "replace", "--langs", "tr"])
    evaluate.main()
    report = json.loads((tmp_path / "results" / "pipeline-fake-ceviri-replace-tr_3.json").read_text(encoding="utf-8"))
    assert set(report["groups"]) == {"synthetic/tr"} and report["groups"]["synthetic/tr"]["hit@1"] == 1.0
    assert report["translation"]["translated_queries"] == 2 and report["translation"]["tokens"] == 84
    assert {e["qid"] for e in report["translation"]["examples"]} == {"s-tr", "o-tr"}
    assert report["guard_accept_rate"] == {"offtopic/tr": 1.0, "synthetic/tr": 1.0}
