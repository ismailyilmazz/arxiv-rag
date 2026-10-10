import json
import threading
import time
from pathlib import Path
from typing import Callable

from app.store import Store

RESTART_MESSAGE = "Sunucu yeniden başladı; bu iş tamamlanamadı. Lütfen tekrar deneyin."


class Worker:
    def __init__(self, store: Store, runner: Callable, output_dir: Path):
        self.store, self.runner, self.output_dir = store, runner, Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.event, self.stopped = threading.Event(), False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.store.fail_unfinished(RESTART_MESSAGE)
        self.thread.start()

    def stop(self) -> None:
        self.stopped = True
        self.event.set()

    def notify(self) -> None:
        self.event.set()

    def _loop(self) -> None:
        while not self.stopped:
            if not self.run_once():
                self.event.wait(timeout=1.0)
                self.event.clear()

    def paths(self, job_id: str) -> tuple[Path, Path]:
        return self.output_dir / f"{job_id}.md", self.output_dir / f"{job_id}.json"

    def run_once(self) -> bool:
        job = self.store.next_queued()
        if not job:
            return False
        job_id = job["id"]
        self.store.update_job(job_id, status="running", started_at=time.time(), stage="cards", done=0, total=0)
        progress = lambda stage, done, total: self.store.update_job(job_id, stage=stage, done=done, total=total)
        try:
            plan = self.store.get_plan(job["plan_id"])
            result = self.runner(plan["selection"], job["removed"], progress)
            md_path, json_path = self.paths(job_id)
            md_path.write_text(result["markdown"], encoding="utf-8")
            report = {k: v for k, v in result.items() if k != "markdown"}
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            summary = {"words": result.get("length", {}).get("words"), "cited": len(result.get("cited", [])),
                       "further_reading": len(result.get("further_reading", [])),
                       "truncated": result.get("length", {}).get("truncated"),
                       "tokens": result.get("tokens", {}).get("total")}
            self.store.update_job(job_id, status="done", stage="done", finished_at=time.time(), summary=summary)
        except Exception as e:
            self.store.update_job(job_id, status="failed", finished_at=time.time(),
                                  error=f"{type(e).__name__}: {e}"[:300])
        return True
