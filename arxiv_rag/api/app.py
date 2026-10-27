import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from arxiv_rag import config
from arxiv_rag.api.schemas import DocumentIn, PlanIn, SearchIn
from arxiv_rag.api.settings import Settings
from arxiv_rag.api.store import Store
from arxiv_rag.api.worker import RESTART_MESSAGE, Worker
from arxiv_rag.generation import document, gemini
from arxiv_rag.storage.arxiv_ids import abs_url
from arxiv_rag.storage.db import connect
from arxiv_rag.storage.deltas import read_summary
from arxiv_rag.storage.tex import clean_authors, clean_tex


def default_pipeline(conn, settings: Settings):
    from arxiv_rag.search.pipeline import SearchPipeline

    return SearchPipeline.load(conn, settings.vectors_dir, settings.models_dir)


def default_selector(conn, pipeline, body: PlanIn, settings: Settings) -> dict:
    return document.select_sources(
        conn,
        pipeline,
        body.topic,
        body.type,
        body.seeds,
        body.lang,
        n_sources=settings.n_sources,
        focus=body.focus,
        force=body.force,
    )


def default_runner(settings: Settings) -> Callable:
    def run(selection: dict, removed: list[str], progress: Callable) -> dict:
        conn, cache = connect(settings.db_path), connect(settings.cache_path)
        try:
            return document.write_from_selection(conn, cache, selection, removed_ids=removed, progress=progress)
        finally:
            conn.close()
            cache.close()

    return run


def document_paths(output_dir: Path, job_id: str) -> tuple[Path, Path]:
    return Path(output_dir) / f"{job_id}.md", Path(output_dir) / f"{job_id}.json"


def log_event(event: str, **fields) -> None:
    line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
    print(json.dumps(line, ensure_ascii=False), file=sys.stderr, flush=True)


def build_info() -> dict:
    try:
        info = json.loads(Path(os.getenv("APP_BUILD_FILE", "build.json")).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"commit": None, "built_at": None}
    return {"commit": info.get("commit") or None, "built_at": info.get("built_at")}


def data_info() -> dict:
    try:
        manifest = json.loads((config.DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"papers": None, "created_at": None}
    applied = read_summary(config.DATA_DIR)
    papers = manifest.get("papers")
    info = {"papers": papers + applied.get("added", 0) if papers else papers, "created_at": manifest.get("created_at")}
    if applied:
        info.update(
            updated_to=max(applied.get("applied") or [None], key=lambda d: d or ""),
            deltas_applied=len(applied.get("applied", [])),
            delta_errors=applied.get("errors", []),
        )
    return info


def create_app(
    settings: Optional[Settings] = None,
    pipeline_factory: Optional[Callable] = None,
    selector: Optional[Callable] = None,
    runner: Optional[Callable] = None,
    start_workers: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_env()
    pipeline_factory, selector = pipeline_factory or default_pipeline, selector or default_selector
    runner = runner or default_runner(settings)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    store = Store(settings.app_db_path)
    shared, planner, lock = {}, {}, threading.Lock()

    def loaded():
        with lock:
            if "pipeline" not in shared:
                shared["conn"] = connect(settings.db_path, check_same_thread=False)
                shared["pipeline"] = pipeline_factory(shared["conn"], settings)
        return shared["conn"], shared["pipeline"]

    def plan_handler(job: dict, progress: Callable) -> dict:
        progress("sources", 0, 1)
        if "pipeline" not in planner:
            _, base = loaded()
            planner["conn"] = connect(settings.db_path)
            planner["pipeline"] = base.bind(planner["conn"])
        selection = selector(planner["conn"], planner["pipeline"], PlanIn(**job["request"]), settings)
        plan_id = store.add_plan(job["ip"], job["request"], selection)
        store.update_job(job["id"], plan_id=plan_id)
        progress("sources", 1, 1)
        return {"plan_id": plan_id, "sources": len(selection["screened_ids"]), "themes": len(selection["themes"])}

    def document_handler(job: dict, progress: Callable) -> dict:
        plan = store.get_plan(job["plan_id"])
        result = runner(plan["selection"], job["request"].get("removed_ids", []), progress)
        md_path, json_path = document_paths(settings.output_dir, job["id"])
        md_path.write_text(result["markdown"], encoding="utf-8")
        report = {k: v for k, v in result.items() if k != "markdown"}
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "words": result.get("length", {}).get("words"),
            "cited": len(result.get("cited", [])),
            "further_reading": len(result.get("further_reading", [])),
            "truncated": result.get("length", {}).get("truncated"),
            "tokens": result.get("tokens", {}).get("total"),
        }

    workers = {
        "plan": Worker(store, "plan", plan_handler),
        "document": Worker(store, "document", document_handler, settings.document_workers),
    }
    app = FastAPI(title="arXiv araştırma asistanı", version="0.7.0")
    app.state.store, app.state.workers, app.state.settings = store, workers, settings
    if start_workers:
        store.fail_unfinished(RESTART_MESSAGE)
        for worker in workers.values():
            worker.start()
    if settings.preload:
        threading.Thread(target=loaded, name="preload", daemon=True).start()

    def client_ip(request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if settings.trust_proxy and forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def enqueue(kind: str, ip: str, request: dict, plan_id: Optional[str] = None) -> dict:
        job_id = store.add_job(kind, ip, request, plan_id)
        workers[kind].notify()
        return {"job_id": job_id, "position": store.position(job_id, workers[kind].concurrency)}

    def job_or_404(job_id: str) -> dict:
        job = store.get_job(job_id)
        if not job:
            raise HTTPException(404, "İş bulunamadı.")
        return job

    api = APIRouter(prefix="/api")

    @api.get("/health")
    def health():
        return {
            "status": "ok",
            "build": build_info(),
            "data": data_info(),
            "pipeline_loaded": "pipeline" in shared,
            "gemini": gemini.status(),
            "queue": {kind: store.active_count(kind) for kind in workers},
            "limits": {
                "plans_per_ip_hour": settings.plans_per_ip_hour,
                "documents_per_ip_day": settings.generations_per_ip_day,
                "documents_per_day": settings.generations_per_day,
            },
        }

    @api.post("/search")
    def search(body: SearchIn):
        conn, pipeline = loaded()
        with lock:
            out = pipeline.search(body.query, k=body.k)
            ids = [r["id"] for r in out["results"]]
            rows = (
                conn.execute(
                    f"SELECT id, title, authors, published, primary_category FROM papers "
                    f"WHERE id IN ({','.join('?' * len(ids))})",
                    ids,
                ).fetchall()
                if ids
                else []
            )
        info = {
            r[0]: {
                "title": clean_tex(r[1]),
                "authors": clean_authors(r[2]),
                "year": (r[3] or "")[:4] or None,
                "category": r[4],
            }
            for r in rows
        }
        results = [
            {"id": r["id"], "url": abs_url(r["id"]), **info.get(r["id"], {}), "score": round(r["score"], 4)}
            for r in out["results"]
        ]
        log_event(
            "search",
            query=body.query[:80],
            accepted=out["accepted"],
            translated=out.get("translated_query"),
            guard_prob=round(out.get("guard_prob") or 0.0, 3),
            results=len(results),
            ms=round(sum((out.get("timings_ms") or {}).values())),
        )
        return {
            "query": body.query,
            "accepted": out["accepted"],
            "guard_prob": out.get("guard_prob"),
            "translated_query": out.get("translated_query"),
            "results": results,
            "timings_ms": out.get("timings_ms", {}),
        }

    @api.post("/plans", status_code=202)
    def create_plan(body: PlanIn, request: Request):
        ip = client_ip(request)
        if store.count_jobs_since("plan", time.time() - 3600, ip) >= settings.plans_per_ip_hour:
            raise HTTPException(429, "Bu saat için plan sınırına ulaştınız. Biraz sonra tekrar deneyin.")
        log_event("plan", topic=body.topic[:80], type=body.type, lang=body.lang, seeds=len(body.seeds))
        return enqueue("plan", ip, body.model_dump())

    @api.get("/plans/{plan_id}")
    def get_plan(plan_id: str):
        plan = store.get_plan(plan_id)
        if not plan:
            raise HTTPException(404, "Plan bulunamadı.")
        view = document.plan_view(plan["selection"])
        view.pop("candidate_ids", None)
        for theme in view["themes"]:
            for paper in theme["papers"]:
                paper["url"] = abs_url(paper["id"])
        for paper in view["further_reading"]:
            paper["url"] = abs_url(paper["id"])
        return {"plan_id": plan_id, **view}

    @api.post("/documents", status_code=202)
    def create_document(body: DocumentIn, request: Request):
        if not store.get_plan(body.plan_id):
            raise HTTPException(404, "Plan bulunamadı.")
        ip, day = client_ip(request), time.time() - 86400
        if store.count_jobs_since("document", day, ip) >= settings.generations_per_ip_day:
            raise HTTPException(429, "Bugünkü üretim hakkınızı kullandınız. Yarın tekrar deneyin.")
        if store.count_jobs_since("document", day) >= settings.generations_per_day:
            raise HTTPException(429, "Sistemin günlük üretim kapasitesi doldu. Yarın tekrar deneyin.")
        log_event("document", plan_id=body.plan_id, removed=len(body.removed_ids))
        return enqueue("document", ip, {"removed_ids": body.removed_ids}, body.plan_id)

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = job_or_404(job_id)
        return {
            "job_id": job_id,
            "kind": job["kind"],
            "status": job["status"],
            "stage": job["stage"],
            "done": job["done"],
            "total": job["total"],
            "position": store.position(job_id, workers[job["kind"]].concurrency),
            "error": job["error"],
            "plan_id": job["plan_id"],
            "summary": job["summary"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
        }

    def finished_document(job_id: str, index: int) -> Path:
        job = job_or_404(job_id)
        if job["kind"] != "document":
            raise HTTPException(404, "Bu iş bir belge üretmiyor.")
        if job["status"] != "done":
            raise HTTPException(409, "Belge henüz hazır değil.")
        return document_paths(settings.output_dir, job_id)[index]

    @api.get("/jobs/{job_id}/document")
    def download_document(job_id: str):
        return FileResponse(
            finished_document(job_id, 0), media_type="text/markdown; charset=utf-8", filename=f"arxiv-rag-{job_id}.md"
        )

    @api.get("/jobs/{job_id}/report")
    def download_report(job_id: str):
        return FileResponse(finished_document(job_id, 1), media_type="application/json")

    app.include_router(api)
    if (Path(settings.frontend_dir) / "index.html").exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app
