"""Makale deposu: SQLite tablosu + FTS5 tam metin arama indeksi.

Neden FTS5: SQLite'ın içinde gelen, BM25 ile sıralama yapan bir arama motoru.
Ek bir servis kurmaya gerek kalmıyor.

Neden porter tokenizer: "networks" ile "network" aynı köke iner.
Aynı tokenizer hem makalelere hem sorgulara uygulandığı için eski projedeki
"eğitimde temizlenip sorguda temizlenmeme" hatası burada oluşamaz.
"""
import sqlite3
from pathlib import Path
from typing import Iterable

COLUMNS = (
    "id", "title", "abstract", "authors", "categories",
    "primary_category", "published", "updated", "license", "doi",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    pk               INTEGER PRIMARY KEY,   -- FTS5 bu sabit sayıya bağlanır
    id               TEXT NOT NULL UNIQUE,  -- arXiv kimliği: 2310.01234 veya hep-th/9901001
    title            TEXT NOT NULL,
    abstract         TEXT NOT NULL,
    authors          TEXT,
    categories       TEXT,                  -- boşlukla ayrılmış: "cs.CL cs.LG"
    primary_category TEXT,
    published        TEXT,                  -- ilk sürüm tarihi, YYYY-MM-DD
    updated          TEXT,
    license          TEXT,
    doi              TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract,
    content='papers', content_rowid='pk',
    tokenize='porter unicode61'
);

-- Tablo değiştikçe arama indeksi kendiliğinden güncel kalsın
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


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def upsert_papers(conn: sqlite3.Connection, rows: Iterable[tuple]) -> None:
    """Satırlar COLUMNS sırasında olmalı. Aynı kimlik varsa kayıt güncellenir.

    Bu fonksiyon hem ilk toplu yüklemede hem de ileride OAI-PMH ile gelen
    günlük güncellemelerde kullanılacak.
    """
    with conn:
        conn.executemany(_UPSERT, rows)
