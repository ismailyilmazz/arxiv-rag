import json
import sys

from core import config, llm
from core.db import connect, init_db, upsert_papers
from core.splits import split_of
from scripts import gen_train_queries
from tests.conftest import make_row


def _setup(tmp_path, monkeypatch, n_train=6):
    ids = [f"2310.{i:05d}" for i in range(400)]
    train_ids = [i for i in ids if split_of(i) == "train"][:n_train]
    test_ids = [i for i in ids if split_of(i) == "test"][:2]
    db = tmp_path / "papers.db"
    conn = connect(db)
    init_db(conn)
    upsert_papers(conn, [make_row(pid, f"Title {pid}", f"Abstract about topic {pid}.") for pid in train_ids + test_ids])
    conn.close()
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(json.dumps({"qid": "m", "type": "manual", "lang": "tr", "query": "x",
                                     "target_id": train_ids[0]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "EVAL_QUERIES_PATH", eval_file)
    monkeypatch.setattr(llm, "_client", lambda: None)
    return db, train_ids, test_ids


def _fake_chat(daily_limit_for=(), calls=None):
    def chat(prompt, model, **kwargs):
        if calls is not None:
            calls.append(model)
        if model in daily_limit_for:
            raise Exception("Rate limit reached for model on tokens per day (TPD): Limit 200000")
        if "search box about" in prompt:
            return {"questions": [f"question {model} {len(prompt)} {i}" for i in range(3)]}, 100
        return {"en": "query about topic", "tr": "konu hakkında sorgu", "mixed": "topic ile ilgili arama"}, 50
    return chat


def _run(db, out, monkeypatch, models):
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--out-dir", str(out), "--n-papers", "10",
                                      "--n-offtopic", "40", "--models", models, "--min-interval", "0"])
    gen_train_queries.main()


def test_generates_three_queries_per_train_paper_and_skips_eval_targets(tmp_path, monkeypatch):
    db, train_ids, test_ids = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(llm, "chat", _fake_chat())
    out = tmp_path / "train"
    _run(db, out, monkeypatch, "a,b")
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    targets = {r["target_id"] for r in rows}
    assert targets == set(train_ids[1:])
    assert not targets & set(test_ids)
    assert len(rows) == 3 * len(targets)
    assert all(r["query"] for r in rows)
    assert {r["type"] for r in rows} == {"synthetic", "mixed"}
    assert (out / "offtopic.jsonl").exists()


def test_switches_model_when_daily_quota_is_hit(tmp_path, monkeypatch):
    db, train_ids, _ = _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(llm, "chat", _fake_chat(daily_limit_for=("a",), calls=calls))
    out = tmp_path / "train"
    _run(db, out, monkeypatch, "a,b")
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["model"] for r in rows} == {"b"}
    assert calls.count("a") == 1


def test_stops_when_all_quotas_are_used_and_resumes_later(tmp_path, monkeypatch):
    db, train_ids, _ = _setup(tmp_path, monkeypatch)
    out = tmp_path / "train"
    monkeypatch.setattr(llm, "chat", _fake_chat(daily_limit_for=("a", "b")))
    _run(db, out, monkeypatch, "a,b")
    assert not (out / "queries.jsonl").exists()

    monkeypatch.setattr(llm, "chat", _fake_chat())
    _run(db, out, monkeypatch, "a,b")
    first = (out / "queries.jsonl").read_text(encoding="utf-8")
    _run(db, out, monkeypatch, "a,b")
    assert (out / "queries.jsonl").read_text(encoding="utf-8") == first


def test_limit_kind():
    assert llm.limit_kind(Exception("Rate limit reached ... tokens per day (TPD)")) == "day"
    assert llm.limit_kind(Exception("Rate limit reached ... tokens per minute (TPM)")) == "minute"
    assert llm.limit_kind(ValueError("bad json")) is None


def test_dry_run_without_offtopic_does_not_block_later_offtopic(tmp_path, monkeypatch):
    db, _, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(llm, "chat", _fake_chat())
    out = tmp_path / "train"
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--out-dir", str(out), "--n-papers", "2",
                                      "--n-offtopic", "0", "--models", "a", "--min-interval", "0"])
    gen_train_queries.main()
    assert not (out / "offtopic.jsonl").exists()
    _run(db, out, monkeypatch, "a")
    assert len((out / "offtopic.jsonl").read_text(encoding="utf-8").splitlines()) > 0


def test_model_with_repeated_minute_limits_is_disabled(tmp_path, monkeypatch):
    db, _, _ = _setup(tmp_path, monkeypatch, n_train=20)
    calls = []

    def chat(prompt, model, **kwargs):
        calls.append(model)
        if model == "q":
            raise Exception("Rate limit reached for model on tokens per minute (TPM): Limit 8000, Requested 41000")
        return {"en": "query about topic", "tr": "konu hakkında sorgu", "mixed": "topic ile ilgili arama"}, 50

    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(gen_train_queries.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gen_train_queries.time, "sleep", sleep)
    out = tmp_path / "train"
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--out-dir", str(out), "--n-papers", "20",
                                      "--n-offtopic", "0", "--models", "q,b", "--min-interval", "10"])
    gen_train_queries.main()
    assert calls.count("q") == gen_train_queries.ModelPool.MAX_STRIKES
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["model"] for r in rows} == {"b"}


def test_copied_or_english_mixed_query_is_dropped(tmp_path, monkeypatch):
    db, train_ids, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(llm, "chat", lambda prompt, model, **kw: (
        {"en": "scale invariant theory", "tr": "ölçekten bağımsız teori", "mixed": "scale invariant theory"}, 50))
    out = tmp_path / "train"
    _run(db, out, monkeypatch, "a")
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["type"] for r in rows} == {"synthetic"}
    assert {r["lang"] for r in rows} == {"en", "tr"}
    dropped = [json.loads(l) for l in (out / "dropped.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(d["key"], d["reason"]) for d in dropped} == {("mixed", "ingilizcenin_tekrari")}
    assert len(dropped) == len(train_ids) - 1


def test_output_token_limit_is_learned_from_error(tmp_path, monkeypatch):
    db, _, _ = _setup(tmp_path, monkeypatch)
    requested = []

    def chat(prompt, model, max_tokens=1024, **kwargs):
        requested.append((model, max_tokens))
        if model == "q" and max_tokens > 1000:
            raise Exception("Request too large ... output tokens per minute (OTPM): Limit 1000, Requested 1024. "
                            "rate limit")
        return {"en": "query about topic", "tr": "konu hakkında sorgu", "mixed": "topic ile ilgili arama"}, 50

    monkeypatch.setattr(llm, "chat", chat)
    out = tmp_path / "train"
    _run(db, out, monkeypatch, "q")
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and {r["model"] for r in rows} == {"q"}
    assert ("q", 250) in requested


def test_model_returning_broken_json_is_disabled_and_paper_goes_to_another(tmp_path, monkeypatch):
    db, train_ids, _ = _setup(tmp_path, monkeypatch)

    def chat(prompt, model, **kwargs):
        if model == "j":
            raise ValueError("Unterminated string")
        return {"en": "query about topic", "tr": "konu hakkında sorgu", "mixed": "topic ile ilgili arama"}, 50

    monkeypatch.setattr(llm, "chat", chat)
    out = tmp_path / "train"
    _run(db, out, monkeypatch, "j,b")
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["target_id"] for r in rows} == set(train_ids[1:])
    assert {r["model"] for r in rows} == {"b"}


def test_offtopic_asks_at_most_25_questions_per_call(tmp_path, monkeypatch):
    db, _, _ = _setup(tmp_path, monkeypatch)
    asked = []

    def chat(prompt, model, **kwargs):
        if "search box about" in prompt:
            asked.append(int(prompt.split()[1]))
            return {"questions": [f"q {len(asked)} {i}" for i in range(asked[-1])]}, 100
        return {"en": "query about topic", "tr": "konu hakkında sorgu", "mixed": "topic ile ilgili arama"}, 50

    monkeypatch.setattr(llm, "chat", chat)
    out = tmp_path / "train"
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--out-dir", str(out), "--n-papers", "1",
                                      "--n-offtopic", "1000", "--models", "a", "--min-interval", "0"])
    gen_train_queries.main()
    assert max(asked) <= 25
    assert len((out / "offtopic.jsonl").read_text(encoding="utf-8").splitlines()) >= 1000


def test_model_disabled_in_offtopic_still_used_for_papers(tmp_path, monkeypatch):
    db, train_ids, _ = _setup(tmp_path, monkeypatch)

    def chat(prompt, model, **kwargs):
        if "search box about" in prompt:
            if model == "q":
                raise ValueError("truncated json")
            return {"questions": [f"{model} q {len(prompt)} {i}" for i in range(5)]}, 100
        return {"en": "query about topic", "tr": "konu hakkında sorgu", "mixed": "topic ile ilgili arama"}, 50

    monkeypatch.setattr(llm, "chat", chat)
    out = tmp_path / "train"
    _run(db, out, monkeypatch, "q,b")
    rows = [json.loads(l) for l in (out / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "q" in {r["model"] for r in rows}


def test_new_words_keeps_turkish_without_special_letters_and_drops_rephrased_english():
    en = "dilaton black hole quasinormal modes sw theory"
    assert gen_train_queries.new_words("dilaton kara delik qnm sw teorisi", en)
    assert not gen_train_queries.new_words(
        "visual hallucination multimodal model detection", "visual hallucination detection in multimodal models")
    assert not gen_train_queries.new_words(
        "curvaton reheating scale invariant two measure theory",
        "curvaton reheating in scale invariant two measure theory")