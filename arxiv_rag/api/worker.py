import threading
import time
from typing import Callable

from arxiv_rag.api.store import Store

RESTART_MESSAGE = "Sunucu yeniden başladı; bu iş tamamlanamadı. Lütfen tekrar deneyin."


class Worker:
    def __init__(self, store: Store, kind: str, handler: Callable):
        self.store, self.kind, self.handler = store, kind, handler
        self.event, self.stopped = threading.Event(), False
        self.thread = threading.Thread(target=self._loop, name=f"worker-{kind}", daemon=True)

    def start(self) -> None:
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

    def run_once(self) -> bool:
        job = self.store.next_queued(self.kind)
        if not job:
            return False
        job_id = job["id"]
        self.store.update_job(job_id, status="running", started_at=time.time(), done=0, total=0)

        def progress(stage, done, total):
            return self.store.update_job(job_id, stage=stage, done=done, total=total)

        try:
            summary = self.handler(job, progress)
            self.store.update_job(job_id, status="done", stage="done", finished_at=time.time(), summary=summary)
        except Exception as e:
            message = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            self.store.update_job(job_id, status="failed", finished_at=time.time(), error=message[:300])
        return True
