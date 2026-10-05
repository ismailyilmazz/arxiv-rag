import re
import sqlite3
from typing import Optional

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"\w+", re.UNICODE)
_stem = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM).stem

_SQL = """
SELECT p.id, -f.rank AS score
FROM (SELECT rowid, rank FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?) AS f
JOIN papers p ON p.pk = f.rowid
ORDER BY f.rank
"""


def query_words(text: str) -> list[str]:
    words = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 1 and w not in ENGLISH_STOP_WORDS and w not in words:
            words.append(w)
    return words


def _any_of(words: list[str]) -> str:
    return " OR ".join(f'"{w}"' for w in words)


def to_fts_query(text: str) -> str:
    return _any_of(query_words(text))


def build_term_df(conn: sqlite3.Connection) -> int:
    with conn:
        conn.execute("DROP TABLE IF EXISTS term_df")
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.papers_vocab USING fts5vocab(main, papers_fts, 'row')")
        conn.execute("CREATE TABLE term_df (term TEXT PRIMARY KEY, doc INTEGER NOT NULL) WITHOUT ROWID")
        conn.execute("INSERT INTO term_df SELECT term, doc FROM temp.papers_vocab")
    return conn.execute("SELECT COUNT(*) FROM term_df").fetchone()[0]


def has_term_df(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'term_df'").fetchone() is not None


def doc_freq(conn: sqlite3.Connection, word: str) -> int:
    row = conn.execute("SELECT doc FROM term_df WHERE term = ?", (_stem(word),)).fetchone()
    return row[0] if row else 0


def gated_query(conn: sqlite3.Connection, text: str, gate: int = 2) -> str:
    words = query_words(text)
    known = sorted((df, w) for w in words if (df := doc_freq(conn, w)) > 0)
    if len(known) <= gate + 1:
        return _any_of(words)
    rare = [w for _, w in known[:gate]]
    rest = [w for w in words if w not in rare]
    return f"({_any_of(rare)}) AND ({_any_of(rest)})"


def search_pks(conn: sqlite3.Connection, text: str, k: int = 10, gate: Optional[int] = None) -> list[tuple[int, float]]:
    query = gated_query(conn, text, gate) if gate else to_fts_query(text)
    if not query:
        return []
    sql = "SELECT rowid, -rank FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?"
    return [(r[0], r[1]) for r in conn.execute(sql, (query, k))]


def search(conn: sqlite3.Connection, text: str, k: int = 10, gate: Optional[int] = None) -> list[tuple[str, float]]:
    query = gated_query(conn, text, gate) if gate else to_fts_query(text)
    if not query:
        return []
    return [(r[0], r[1]) for r in conn.execute(_SQL, (query, k))]
