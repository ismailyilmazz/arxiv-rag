import threading
import time
from typing import Callable

from arxiv_rag.api.store import Store

RESTART_MESSAGE = "Sunucu yeniden başladı; bu iş tamamlanamadı. Lütfen tekrar deneyin."


class Worker:
    def __init__(self, store: Store, kind: str, handler: Callable, concurrency: int = 1):
        self.store, self.kind, self.handler = store, kind, handler
        self.concurrency = max(1, concurrency)
        self.event, self.stopped = threading.Event(), False
        self.threads = [
            threading.Thread(target=self._loop, name=f"worker-{kind}-{i}", daemon=True) for i in range(self.concurrency)
        ]

    def start(self) -> None:
        for thread in self.threads:
            thread.start()

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
        job = self.store.claim_next(self.kind)
        if not job:
            return False
        job_id = job["id"]

        def progress(stage, done, total):
            return self.store.update_job(job_id, stage=stage, done=done, total=total)

        try:
            summary = self.handler(job, progress)
            self.store.update_job(job_id, status="done", stage="done", finished_at=time.time(), summary=summary)
        except Exception as e:
            message = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            self.store.update_job(job_id, status="failed", finished_at=time.time(), error=message[:300])
        return True
