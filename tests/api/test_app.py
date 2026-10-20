import json
import sqlite3
import time

from fastapi.testclient import TestClient

from arxiv_rag.api.app import create_app
from arxiv_rag.api.settings import Settings
from arxiv_rag.api.store import Store
from arxiv_rag.api.worker import RESTART_MESSAGE
from arxiv_rag.storage.db import connect, init_db, upsert_papers
from tests.conftest import make_row

IDS = ["2101.03961", "1701.06538", "2401.04088"]


class FakeSearch:
    def __init__(self):
        self.bound = []

    def search(self, text, k=10):
        return {
            "results": [{"id": pid, "score": 1.0 - i / 10} for i, pid in enumerate(IDS[:k])],
            "accepted": True,
            "guard_prob": 0.8,
            "timings_ms": {"total_ms": 1.0},
        }

    def bind(self, conn):
        self.bound.append(conn)
        return self


def selection(topic="experts", doc_type="survey", lang="en"):
    return {
        "version": 1,
        "topic": topic,
        "focus": None,
        "doc_type": doc_type,
        "lang": lang,
        "subject": topic,
        "seeds": [],
        "queries": [topic],
        "rejected_queries": [],
        "forced": False,
        "scholar": {"ok": True},
        "candidate_ids": IDS + ["9999.00001"],
        "off_topic_candidates": 1,
        "screened_ids": IDS,
        "surveys_moved": ["2312.10997"],
        "dropped_by_themes": [],
        "backfilled": [],
        "deep": [IDS[0]],
        "themes": [{"name": "Routing", "description": "", "ids": IDS}],
        "relevance": {pid: 0.95 for pid in IDS},
        "titles": {pid: f"Title {pid}" for pid in IDS + ["2312.10997"]},
        "tokens": {"expand": 10, "themes": 20},
        "timings_ms": {"retrieval_ms": 5},
    }


class Recorder:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, sel, removed, progress):
        self.calls.append((sel["topic"], list(removed)))
        progress("sections", 3, 3)
        if self.fail:
            raise RuntimeError("model unavailable")
        return {
            "markdown": "# Başlık\n\nMetin [1].\n",
            "length": {"words": 3, "truncated": False},
            "cited": [IDS[0]],
            "further_reading": [],
            "tokens": {"total": 42},
            "removed_ids": list(removed),
        }


def build(tmp_path, runner=None, selector=None, start_workers=False, **extra):
    conn = connect(tmp_path / "papers.db")
    init_db(conn)
    upsert_papers(conn, [make_row(pid, f"Paper {pid}", "Abstract.") for pid in IDS])
    conn.close()
    settings = Settings(
        db_path=tmp_path / "papers.db",
        cache_path=tmp_path / "cache.db",
        app_db_path=tmp_path / "app.db",
        output_dir=tmp_path / "jobs",
        frontend_dir=tmp_path / "frontend",
        **extra,
    )
    fake = FakeSearch()
    app = create_app(
        settings,
        pipeline_factory=lambda conn, s: fake,
        selector=selector or (lambda conn, pipeline, body, s: selection(body.topic, body.type, body.lang)),
        runner=runner or Recorder(),
        start_workers=start_workers,
    )
    return app, TestClient(app), fake


def run(app, kind):
    assert app.state.workers[kind].run_once() is True


def test_health_and_search(tmp_path):
    app, client, _ = build(tmp_path)
    assert client.get("/api/health").json()["pipeline_loaded"] is False
    out = client.post("/api/search", json={"query": "mixture of experts", "k": 2}).json()
    assert out["accepted"] and [r["id"] for r in out["results"]] == IDS[:2]
    assert out["results"][0]["title"] == f"Paper {IDS[0]}" and out["results"][0]["url"].endswith(IDS[0])
    health = client.get("/api/health").json()
    assert health["pipeline_loaded"] and health["queue"] == {"plan": 0, "document": 0}
    assert health["gemini"] == {"enabled": False}
    assert client.post("/api/search", json={"query": "x"}).status_code == 422


def test_plan_then_document_flow(tmp_path):
    runner = Recorder()
    app, client, fake = build(tmp_path, runner=runner)
    created = client.post("/api/plans", json={"topic": "mixture of experts", "type": "proposal", "lang": "tr"})
    assert created.status_code == 202 and created.json()["position"] == 1
    job_id = created.json()["job_id"]
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "queued"
    run(app, "plan")
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["kind"] == "plan" and job["status"] == "done" and job["summary"]["sources"] == 3
    assert len(fake.bound) == 1
    plan = client.get(f"/api/plans/{job['plan_id']}").json()
    assert plan["doc_type"] == "proposal" and "candidate_ids" not in plan
    assert plan["themes"][0]["papers"][0]["url"].endswith(IDS[0])
    assert plan["further_reading"][0]["url"].endswith("2312.10997")

    doc = client.post("/api/documents", json={"plan_id": job["plan_id"], "removed_ids": [IDS[2]]})
    assert doc.status_code == 202
    doc_id = doc.json()["job_id"]
    assert client.get(f"/api/jobs/{doc_id}/document").status_code == 409
    run(app, "document")
    status = client.get(f"/api/jobs/{doc_id}").json()
    assert status["status"] == "done" and status["summary"]["tokens"] == 42 and status["plan_id"] == job["plan_id"]
    assert runner.calls == [("mixture of experts", [IDS[2]])]
    download = client.get(f"/api/jobs/{doc_id}/document")
    assert download.text.startswith("# Başlık") and "attachment" in download.headers["content-disposition"]
    assert client.get(f"/api/jobs/{doc_id}/report").json()["removed_ids"] == [IDS[2]]
    assert client.get(f"/api/jobs/{job_id}/document").status_code == 404


def test_failures_have_readable_messages(tmp_path):
    def no_sources(conn, pipeline, body, settings):
        raise ValueError("Bu konu için yeterli makale bulunamadı.")

    app, client, _ = build(tmp_path, selector=no_sources, runner=Recorder(fail=True))
    job_id = client.post("/api/plans", json={"topic": "pizza"}).json()["job_id"]
    run(app, "plan")
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed" and job["error"] == "Bu konu için yeterli makale bulunamadı."

    plan_id = app.state.store.add_plan("ip", {"topic": "x"}, selection())
    doc_id = client.post("/api/documents", json={"plan_id": plan_id}).json()["job_id"]
    run(app, "document")
    assert client.get(f"/api/jobs/{doc_id}").json()["error"] == "RuntimeError: model unavailable"


def test_missing_and_invalid(tmp_path):
    app, client, _ = build(tmp_path)
    assert client.get("/api/plans/nope").status_code == 404
    assert client.post("/api/documents", json={"plan_id": "nope"}).status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/plans", json={"topic": "x", "type": "poem"}).status_code == 422
    assert client.post("/api/plans", json={"topic": "experts", "seeds": IDS * 2}).status_code == 422


def test_limits(tmp_path):
    app, client, _ = build(
        tmp_path, plans_per_ip_hour=2, generations_per_ip_day=1, generations_per_day=2, trust_proxy=True
    )
    client.post("/api/plans", json={"topic": "experts"})
    client.post("/api/plans", json={"topic": "experts"})
    over = client.post("/api/plans", json={"topic": "experts"})
    assert over.status_code == 429 and "plan sınırı" in over.json()["detail"]

    plan_id = app.state.store.add_plan("ip", {"topic": "x"}, selection())

    def post(ip):
        return client.post("/api/documents", json={"plan_id": plan_id}, headers={"x-forwarded-for": ip})

    assert post("10.0.0.1").status_code == 202
    again = post("10.0.0.1")
    assert again.status_code == 429 and "hakkınızı" in again.json()["detail"]
    assert post("10.0.0.2").status_code == 202
    full = post("10.0.0.3")
    assert full.status_code == 429 and "kapasitesi" in full.json()["detail"]


def test_restart_marks_running_jobs_failed_and_old_schema_is_kept(tmp_path):
    old = sqlite3.connect(tmp_path / "app.db")
    old.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, status TEXT)")
    old.execute("INSERT INTO jobs VALUES ('x', 'p', 'done')")
    old.commit()
    old.close()
    store = Store(tmp_path / "app.db")
    assert store.conn.execute("SELECT COUNT(*) FROM jobs_v1").fetchone()[0] == 1
    job_id = store.add_job("document", "ip", {}, "p")
    store.update_job(job_id, status="running")
    app, client, _ = build(tmp_path, start_workers=True)
    try:
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "failed" and job["error"] == RESTART_MESSAGE
    finally:
        for worker in app.state.workers.values():
            worker.stop()


def test_background_workers_end_to_end(tmp_path):
    app, client, _ = build(tmp_path, start_workers=True)

    def wait(job_id):
        deadline = time.time() + 5
        while time.time() < deadline:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("done", "failed"):
                return job
            time.sleep(0.05)
        raise AssertionError("iş bitmedi")

    try:
        plan_job = wait(client.post("/api/plans", json={"topic": "experts"}).json()["job_id"])
        doc_job = wait(client.post("/api/documents", json={"plan_id": plan_job["plan_id"]}).json()["job_id"])
        assert doc_job["status"] == "done"
        assert client.get(f"/api/jobs/{doc_job['job_id']}/document").text.startswith("# Başlık")
    finally:
        for worker in app.state.workers.values():
            worker.stop()


def test_frontend_is_served_when_built(tmp_path):
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "index.html").write_text("<html>arxiv-rag</html>", encoding="utf-8")
    app, client, _ = build(tmp_path)
    assert "arxiv-rag" in client.get("/").text
    assert client.get("/api/health").json()["status"] == "ok"


def test_preload_loads_the_pipeline_at_startup(tmp_path):
    app, client, _ = build(tmp_path, preload=True)
    deadline = time.time() + 5
    while time.time() < deadline and not client.get("/api/health").json()["pipeline_loaded"]:
        time.sleep(0.02)
    assert client.get("/api/health").json()["pipeline_loaded"] is True


def test_search_cleans_tex_titles_and_logs_one_line(tmp_path, capsys):
    app, client, _ = build(tmp_path)
    conn = connect(tmp_path / "papers.db")
    conn.execute(
        "UPDATE papers SET title = ?, authors = ? WHERE id = ?", (r"T\"{u}rk\c{c}e veri", r"A. G\"{o}k", IDS[0])
    )
    conn.commit()
    conn.close()
    capsys.readouterr()
    out = client.post("/api/search", json={"query": "mixture of experts", "k": 2}).json()
    assert out["results"][0]["title"] == "Türkçe veri" and out["results"][0]["authors"] == "A. Gök"
    lines = [json.loads(line) for line in capsys.readouterr().err.splitlines() if line.startswith("{")]
    assert [line["event"] for line in lines] == ["search"]
    assert lines[0]["query"] == "mixture of experts" and lines[0]["results"] == 2
