import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY, created_at REAL NOT NULL, ip TEXT NOT NULL, request TEXT NOT NULL, selection TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL, finished_at REAL,
    ip TEXT NOT NULL, status TEXT NOT NULL, stage TEXT, done INTEGER, total INTEGER, removed TEXT NOT NULL,
    error TEXT, summary TEXT);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, created_at);
"""


class Store:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def _one(self, sql: str, args=()) -> Optional[dict]:
        with self.lock:
            row = self.conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _write(self, sql: str, args=()) -> None:
        with self.lock:
            self.conn.execute(sql, args)
            self.conn.commit()

    def add_plan(self, ip: str, request: dict, selection: dict) -> str:
        plan_id = uuid.uuid4().hex[:12]
        self._write("INSERT INTO plans VALUES (?, ?, ?, ?, ?)",
                    (plan_id, time.time(), ip, json.dumps(request), json.dumps(selection)))
        return plan_id

    def get_plan(self, plan_id: str) -> Optional[dict]:
        row = self._one("SELECT * FROM plans WHERE id = ?", (plan_id,))
        if row:
            row["request"], row["selection"] = json.loads(row["request"]), json.loads(row["selection"])
        return row

    def add_job(self, plan_id: str, ip: str, removed: list[str]) -> str:
        job_id = uuid.uuid4().hex[:12]
        self._write("INSERT INTO jobs (id, plan_id, created_at, ip, status, removed) VALUES (?, ?, ?, ?, 'queued', ?)",
                    (job_id, plan_id, time.time(), ip, json.dumps(removed)))
        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        row = self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row:
            row["removed"] = json.loads(row["removed"])
            row["summary"] = json.loads(row["summary"]) if row["summary"] else None
        return row

    def update_job(self, job_id: str, **fields) -> None:
        if "summary" in fields and fields["summary"] is not None:
            fields["summary"] = json.dumps(fields["summary"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._write(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def next_queued(self) -> Optional[dict]:
        row = self._one("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1")
        return self.get_job(row["id"]) if row else None

    def position(self, job_id: str) -> int:
        job = self._one("SELECT created_at, status FROM jobs WHERE id = ?", (job_id,))
        if not job or job["status"] != "queued":
            return 0
        ahead = self._one("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running') AND created_at < ?",
                          (job["created_at"],))
        return ahead["n"] + 1

    def queued_count(self) -> int:
        return self._one("SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running')")["n"]

    def count_jobs_since(self, since: float, ip: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM jobs WHERE created_at >= ? AND status != 'failed'"
        args = [since]
        if ip is not None:
            sql, args = sql + " AND ip = ?", args + [ip]
        return self._one(sql, args)["n"]

    def count_plans_since(self, since: float, ip: str) -> int:
        return self._one("SELECT COUNT(*) AS n FROM plans WHERE created_at >= ? AND ip = ?", (since, ip))["n"]

    def fail_unfinished(self, message: str) -> int:
        with self.lock:
            cur = self.conn.execute("UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE status = 'running'",
                                    (message, time.time()))
            self.conn.commit()
        return cur.rowcount
