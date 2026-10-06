import sqlite3
from pathlib import Path
from typing import Iterable

COLUMNS = (
    "id", "title", "abstract", "authors", "categories",
    "primary_category", "published", "updated", "license", "doi",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    pk               INTEGER PRIMARY KEY,
    id               TEXT NOT NULL UNIQUE,
    title            TEXT NOT NULL,
    abstract         TEXT NOT NULL,
    authors          TEXT,
    categories       TEXT,
    primary_category TEXT,
    published        TEXT,
    updated          TEXT,
    license          TEXT,
    doi              TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract,
    content='papers', content_rowid='pk',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract) VALUES (new.pk, new.title, new.abstract);
END;
CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
    VALUES ('delete', old.pk, old.title, old.abstract);
END;
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract)
    VALUES ('delete', old.pk, old.title, old.abstract);
    INSERT INTO papers_fts(rowid, title, abstract) VALUES (new.pk, new.title, new.abstract);
END;
"""

_UPSERT = f"""
INSERT INTO papers ({", ".join(COLUMNS)})
VALUES ({", ".join("?" for _ in COLUMNS)})
ON CONFLICT(id) DO UPDATE SET
    {", ".join(f"{c}=excluded.{c}" for c in COLUMNS if c != "id")}
"""


def connect(path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def upsert_papers(conn: sqlite3.Connection, rows: Iterable[tuple]) -> None:
    with conn:
        conn.executemany(_UPSERT, rows)