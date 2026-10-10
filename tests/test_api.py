import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import Settings
from app.worker import RESTART_MESSAGE, Worker
from core.db import connect, init_db, upsert_papers
from tests.conftest import make_row

IDS = ["2101.03961", "1701.06538", "2401.04088"]


class FakeSearch:
    def __init__(self, accepted=True):
        self.accepted, self.queries = accepted, []

    def search(self, text, k=10):
        self.queries.append(text)
        return {"results": [{"id": pid, "score": 1.0 - i / 10, "dense_score": 0.9} for i, pid in enumerate(IDS[:k])],
                "accepted": self.accepted, "guard_prob": 0.8, "timings_ms": {"total_ms": 1.0}}


def selection(topic="experts", doc_type="survey", lang="en"):
    return {"version": 1, "topic": topic, "focus": None, "doc_type": doc_type, "lang": lang, "subject": topic,
            "seeds": [], "queries": [topic], "rejected_queries": [], "forced": False, "scholar": {"ok": True},
            "candidate_ids": IDS + ["9999.00001"], "off_topic_candidates": 1, "screened_ids": IDS,
            "surveys_moved": [], "dropped_by_themes": [], "backfilled": [], "deep": [IDS[0]],
            "themes": [{"name": "Routing", "description": "", "ids": IDS}],
            "relevance": {pid: 0.95 for pid in IDS}, "titles": {pid: f"Title {pid}" for pid in IDS},
            "tokens": {"expand": 10, "themes": 20}, "timings_ms": {"retrieval_ms": 5}}


class Recorder:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, selection, removed, progress):
        self.calls.append((selection["topic"], list(removed)))
        progress("cards", 1, 1)
        progress("sections", 3, 3)
        if self.fail:
            raise RuntimeError("model unavailable")
        return {"markdown": "# Başlık\n\nMetin [1].\n", "length": {"words": 3, "truncated": False},
                "cited": [IDS[0]], "further_reading": [], "tokens": {"total": 42}, "removed_ids": list(removed)}


def build(tmp_path, runner=None, selector=None, start_worker=False, accepted=True, **limits):
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(conn, [make_row(pid, f"Paper {pid}", "Abstract.") for pid in IDS])
    conn.close()
    settings = Settings(db_path=tmp_path / "papers.db", cache_path=tmp_path / "cache.db",
                        app_db_path=tmp_path / "app.db", output_dir=tmp_path / "jobs", **limits)
    fake = FakeSearch(accepted)
    app = create_app(settings, pipeline_factory=lambda conn, s: fake,
                     selector=selector or (lambda conn, pipeline, body, s: selection(body.topic, body.type, body.lang)),
                     runner=runner or Recorder(), start_worker=start_worker)
    return app, TestClient(app), fake


def test_health_and_search(tmp_path):
    app, client, fake = build(tmp_path)
    assert client.get("/health").json()["pipeline_loaded"] is False
    out = client.post("/search", json={"query": "mixture of experts", "k": 2}).json()
    assert out["accepted"] and [r["id"] for r in out["results"]] == IDS[:2]
    first = out["results"][0]
    assert first["title"] == f"Paper {IDS[0]}" and first["url"].endswith(IDS[0]) and "year" in first
    assert client.get("/health").json()["pipeline_loaded"] is True
    assert client.post("/search", json={"query": "x"}).status_code == 422


def test_plan_generate_progress_and_download(tmp_path):
    runner = Recorder()
    app, client, _ = build(tmp_path, runner=runner)
    plan = client.post("/plan", json={"topic": "mixture of experts", "type": "proposal", "lang": "tr"}).json()
    assert plan["dry_run"] and plan["doc_type"] == "proposal" and "candidate_ids" not in plan
    assert plan["themes"][0]["papers"][0]["url"].endswith(IDS[0]) and plan["screened"] == 3

    job = client.post("/generate", json={"plan_id": plan["plan_id"], "removed_ids": [IDS[2]]}).json()
    assert job["position"] == 1
    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "queued"
    assert client.get(f"/jobs/{job['job_id']}/download").status_code == 409

    assert app.state.worker.run_once() is True
    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "done" and status["stage"] == "done" and status["summary"]["tokens"] == 42
    assert runner.calls == [("mixture of experts", [IDS[2]])]
    download = client.get(f"/jobs/{job['job_id']}/download")
    assert download.text.startswith("# Başlık") and "attachment" in download.headers["content-disposition"]
    report = client.get(f"/jobs/{job['job_id']}/report").json()
    assert report["removed_ids"] == [IDS[2]] and "markdown" not in report


def test_errors_and_missing_things(tmp_path):
    def broken(conn, pipeline, body, settings):
        raise ValueError("Bu konu için yeterli makale bulunamadı.")

    app, client, _ = build(tmp_path, selector=broken)
    response = client.post("/plan", json={"topic": "pizza"})
    assert response.status_code == 422 and "yeterli makale" in response.json()["detail"]
    assert client.post("/generate", json={"plan_id": "nope"}).status_code == 404
    assert client.get("/jobs/nope").status_code == 404
    assert client.post("/plan", json={"topic": "x", "type": "poem"}).status_code == 422


def test_failed_job_reports_error(tmp_path):
    app, client, _ = build(tmp_path, runner=Recorder(fail=True))
    plan = client.post("/plan", json={"topic": "experts"}).json()
    job = client.post("/generate", json={"plan_id": plan["plan_id"]}).json()
    app.state.worker.run_once()
    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "failed" and "model unavailable" in status["error"]


def test_limits(tmp_path):
    app, client, _ = build(tmp_path, plans_per_ip_hour=2, generations_per_ip_day=1, generations_per_day=2,
                           trust_proxy=True)
    plan = client.post("/plan", json={"topic": "experts"}).json()
    client.post("/plan", json={"topic": "experts"})
    over = client.post("/plan", json={"topic": "experts"})
    assert over.status_code == 429 and "plan sınırı" in over.json()["detail"]

    a = {"x-forwarded-for": "10.0.0.1"}
    assert client.post("/generate", json={"plan_id": plan["plan_id"]}, headers=a).status_code == 200
    again = client.post("/generate", json={"plan_id": plan["plan_id"]}, headers=a)
    assert again.status_code == 429 and "hakkınızı" in again.json()["detail"]
    assert client.post("/generate", json={"plan_id": plan["plan_id"]},
                       headers={"x-forwarded-for": "10.0.0.2"}).status_code == 200
    full = client.post("/generate", json={"plan_id": plan["plan_id"]}, headers={"x-forwarded-for": "10.0.0.3"})
    assert full.status_code == 429 and "kapasitesi" in full.json()["detail"]


def test_restart_marks_running_jobs_failed(tmp_path):
    app, client, _ = build(tmp_path)
    plan = client.post("/plan", json={"topic": "experts"}).json()
    job = client.post("/generate", json={"plan_id": plan["plan_id"]}).json()
    store = app.state.store
    store.update_job(job["job_id"], status="running")
    restarted = Worker(store, Recorder(), tmp_path / "jobs")
    restarted.start()
    restarted.stop()
    status = client.get(f"/jobs/{job['job_id']}").json()
    assert status["status"] == "failed" and status["error"] == RESTART_MESSAGE


def test_background_worker_end_to_end(tmp_path):
    app, client, _ = build(tmp_path, start_worker=True)
    try:
        plan = client.post("/plan", json={"topic": "experts"}).json()
        job = client.post("/generate", json={"plan_id": plan["plan_id"]}).json()
        deadline = time.time() + 5
        while time.time() < deadline:
            status = client.get(f"/jobs/{job['job_id']}").json()
            if status["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert status["status"] == "done"
        assert client.get(f"/jobs/{job['job_id']}/download").text.startswith("# Başlık")
    finally:
        app.state.worker.stop()


def test_force_flag_reaches_selector(tmp_path):
    seen = {}

    def selector(conn, pipeline, body, settings):
        seen["force"], seen["seeds"] = body.force, body.seeds
        return selection()

    app, client, _ = build(tmp_path, selector=selector)
    client.post("/plan", json={"topic": "experts", "force": True, "seeds": [IDS[0]]})
    assert seen == {"force": True, "seeds": [IDS[0]]}
    too_many = client.post("/plan", json={"topic": "experts", "seeds": IDS * 2})
    assert too_many.status_code == 422
