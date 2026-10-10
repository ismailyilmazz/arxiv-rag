import threading
import time
from typing import Callable, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.settings import Settings
from app.store import Store
from app.worker import Worker
from core import survey
from core.arxiv_ids import abs_url
from core.db import connect


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(10, ge=1, le=30)


class PlanIn(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    type: Literal["survey", "proposal", "synthesis"] = "survey"
    lang: Literal["en", "tr"] = "en"
    seeds: list[str] = Field(default_factory=list, max_length=5)
    focus: Optional[str] = Field(None, max_length=100)
    force: bool = False


class GenerateIn(BaseModel):
    plan_id: str = Field(min_length=1, max_length=40)
    removed_ids: list[str] = Field(default_factory=list, max_length=30)


def default_pipeline(conn, settings: Settings):
    from core.pipeline import SearchPipeline
    return SearchPipeline.load(conn, settings.vectors_dir, settings.models_dir)


def default_selector(conn, pipeline, body: PlanIn, settings: Settings) -> dict:
    return survey.select_sources(conn, pipeline, body.topic, body.type, body.seeds, body.lang,
                                 n_sources=settings.n_sources, focus=body.focus, force=body.force)


def default_runner(settings: Settings) -> Callable:
    def run(selection: dict, removed: list[str], progress: Callable) -> dict:
        conn, cache = connect(settings.db_path), connect(settings.cache_path)
        try:
            return survey.write_from_selection(conn, cache, selection, removed_ids=removed, progress=progress)
        finally:
            conn.close()
            cache.close()
    return run


def create_app(settings: Optional[Settings] = None, pipeline_factory: Optional[Callable] = None,
               selector: Optional[Callable] = None, runner: Optional[Callable] = None,
               start_worker: bool = True) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline_factory = pipeline_factory or default_pipeline
    selector = selector or default_selector
    store = Store(settings.app_db_path)
    worker = Worker(store, runner or default_runner(settings), settings.output_dir)
    state, lock = {}, threading.Lock()
    app = FastAPI(title="arXiv araştırma asistanı", version="0.6.0")
    app.state.store, app.state.worker, app.state.settings = store, worker, settings
    if start_worker:
        worker.start()

    def loaded():
        if "pipeline" not in state:
            state["conn"] = connect(settings.db_path, check_same_thread=False)
            state["pipeline"] = pipeline_factory(state["conn"], settings)
        return state["conn"], state["pipeline"]

    def client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if settings.trust_proxy and forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def papers(conn, ids: list[str]) -> dict:
        if not ids:
            return {}
        rows = conn.execute(f"SELECT id, title, authors, published, primary_category FROM papers "
                            f"WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
        return {r[0]: {"id": r[0], "title": r[1], "authors": r[2], "year": (r[3] or "")[:4] or None,
                       "category": r[4], "url": abs_url(r[0])} for r in rows}

    @app.get("/health")
    def health():
        return {"status": "ok", "pipeline_loaded": "pipeline" in state, "queue": store.queued_count(),
                "limits": {"plans_per_ip_hour": settings.plans_per_ip_hour,
                           "generations_per_ip_day": settings.generations_per_ip_day,
                           "generations_per_day": settings.generations_per_day}}

    @app.post("/search")
    def search(body: SearchIn):
        with lock:
            conn, pipeline = loaded()
            out = pipeline.search(body.query, k=body.k)
            info = papers(conn, [r["id"] for r in out["results"]])
        results = [{**info.get(r["id"], {"id": r["id"], "url": abs_url(r["id"])}), "score": round(r["score"], 4)}
                   for r in out["results"]]
        return {"query": body.query, "accepted": out["accepted"], "guard_prob": out.get("guard_prob"),
                "results": results, "timings_ms": out.get("timings_ms", {})}

    @app.post("/plan")
    def plan(body: PlanIn, request: Request):
        ip = client_ip(request)
        if store.count_plans_since(time.time() - 3600, ip) >= settings.plans_per_ip_hour:
            raise HTTPException(429, "Bu saat için plan sınırına ulaştınız. Biraz sonra tekrar deneyin.")
        with lock:
            conn, pipeline = loaded()
            try:
                selection = selector(conn, pipeline, body, settings)
            except ValueError as e:
                raise HTTPException(422, str(e))
        plan_id = store.add_plan(ip, body.model_dump(), selection)
        view = survey.plan_view(selection)
        for theme in view["themes"]:
            for paper in theme["papers"]:
                paper["url"] = abs_url(paper["id"])
        view.pop("candidate_ids", None)
        return {"plan_id": plan_id, **view}

    @app.post("/generate")
    def generate(body: GenerateIn, request: Request):
        if not store.get_plan(body.plan_id):
            raise HTTPException(404, "Plan bulunamadı.")
        ip, day = client_ip(request), time.time() - 86400
        if store.count_jobs_since(day, ip) >= settings.generations_per_ip_day:
            raise HTTPException(429, "Bugünkü üretim hakkınızı kullandınız. Yarın tekrar deneyin.")
        if store.count_jobs_since(day) >= settings.generations_per_day:
            raise HTTPException(429, "Sistemin günlük üretim kapasitesi doldu. Yarın tekrar deneyin.")
        job_id = store.add_job(body.plan_id, ip, body.removed_ids)
        worker.notify()
        return {"job_id": job_id, "position": store.position(job_id)}

    def job_or_404(job_id: str) -> dict:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "İş bulunamadı.")
        return job

    @app.get("/jobs/{job_id}")
    def job(job_id: str):
        job = job_or_404(job_id)
        return {"job_id": job_id, "status": job["status"], "stage": job["stage"], "done": job["done"],
                "total": job["total"], "position": store.position(job_id), "error": job["error"],
                "summary": job["summary"], "created_at": job["created_at"], "started_at": job["started_at"],
                "finished_at": job["finished_at"]}

    def finished_file(job_id: str, index: int):
        job = job_or_404(job_id)
        if job["status"] != "done":
            raise HTTPException(409, "İş henüz tamamlanmadı.")
        return worker.paths(job_id)[index]

    @app.get("/jobs/{job_id}/download")
    def download(job_id: str):
        return FileResponse(finished_file(job_id, 0), media_type="text/markdown; charset=utf-8",
                            filename=f"arxiv-rag-{job_id}.md")

    @app.get("/jobs/{job_id}/report")
    def report(job_id: str):
        return FileResponse(finished_file(job_id, 1), media_type="application/json")

    return app
