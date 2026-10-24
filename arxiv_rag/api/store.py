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
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at REAL NOT NULL, started_at REAL, finished_at REAL,
    ip TEXT NOT NULL, status TEXT NOT NULL, stage TEXT, done INTEGER, total INTEGER, request TEXT NOT NULL,
    plan_id TEXT, error TEXT, summary TEXT);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (kind, status, created_at);
"""
ACTIVE = ("queued", "running")


class Store:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            columns = {r[1] for r in self.conn.execute("PRAGMA table_info(jobs)")}
            if columns and "kind" not in columns:
                self.conn.execute("ALTER TABLE jobs RENAME TO jobs_v1")
                self.conn.execute("DROP INDEX IF EXISTS jobs_status")
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def _one(self, sql: str, args=()) -> Optional[dict]:
        with self.lock:
            row = self.conn.execute(sql, args).fetchone()
        return dict(row) if row else None

    def _write(self, sql: str, args=()) -> int:
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
        return cur.rowcount

    def add_plan(self, ip: str, request: dict, selection: dict) -> str:
        plan_id = uuid.uuid4().hex[:12]
        self._write(
            "INSERT INTO plans VALUES (?, ?, ?, ?, ?)",
            (plan_id, time.time(), ip, json.dumps(request), json.dumps(selection)),
        )
        return plan_id

    def get_plan(self, plan_id: str) -> Optional[dict]:
        row = self._one("SELECT * FROM plans WHERE id = ?", (plan_id,))
        if row:
            row["request"], row["selection"] = json.loads(row["request"]), json.loads(row["selection"])
        return row

    def add_job(self, kind: str, ip: str, request: dict, plan_id: Optional[str] = None) -> str:
        job_id = uuid.uuid4().hex[:12]
        self._write(
            "INSERT INTO jobs (id, kind, created_at, ip, status, request, plan_id) VALUES (?, ?, ?, ?, 'queued', ?, ?)",
            (job_id, kind, time.time(), ip, json.dumps(request), plan_id),
        )
        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        row = self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row:
            row["request"] = json.loads(row["request"])
            row["summary"] = json.loads(row["summary"]) if row["summary"] else None
        return row

    def update_job(self, job_id: str, **fields) -> None:
        if fields.get("summary") is not None:
            fields["summary"] = json.dumps(fields["summary"])
        self._write(f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?", (*fields.values(), job_id))

    def next_queued(self, kind: str) -> Optional[dict]:
        row = self._one("SELECT id FROM jobs WHERE kind = ? AND status = 'queued' ORDER BY created_at LIMIT 1", (kind,))
        return self.get_job(row["id"]) if row else None

    def claim_next(self, kind: str) -> Optional[dict]:
        with self.lock:
            row = self.conn.execute(
                "SELECT id FROM jobs WHERE kind = ? AND status = 'queued' ORDER BY created_at LIMIT 1", (kind,)
            ).fetchone()
            if not row:
                return None
            self.conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ?, done = 0, total = 0 WHERE id = ?",
                (time.time(), row["id"]),
            )
            self.conn.commit()
        return self.get_job(row["id"])

    def position(self, job_id: str, workers: int = 1) -> int:
        job = self._one("SELECT kind, created_at, status FROM jobs WHERE id = ?", (job_id,))
        if not job or job["status"] != "queued":
            return 0
        ahead = self._one(
            "SELECT COUNT(*) AS n FROM jobs WHERE kind = ? AND status IN (?, ?) AND created_at < ?",
            (job["kind"], *ACTIVE, job["created_at"]),
        )
        return max(1, ahead["n"] + 2 - max(1, workers))

    def active_count(self, kind: str) -> int:
        return self._one("SELECT COUNT(*) AS n FROM jobs WHERE kind = ? AND status IN (?, ?)", (kind, *ACTIVE))["n"]

    def count_jobs_since(self, kind: str, since: float, ip: Optional[str] = None) -> int:
        sql, args = (
            "SELECT COUNT(*) AS n FROM jobs WHERE kind = ? AND created_at >= ? AND status != 'failed'",
            [kind, since],
        )
        if ip is not None:
            sql, args = sql + " AND ip = ?", args + [ip]
        return self._one(sql, args)["n"]

    def fail_unfinished(self, message: str) -> int:
        return self._write(
            "UPDATE jobs SET status = 'failed', error = ?, finished_at = ? WHERE status = 'running'",
            (message, time.time()),
        )
