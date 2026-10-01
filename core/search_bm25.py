import re
import sqlite3

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"\w+", re.UNICODE)

_SQL = """
SELECT p.id, -f.rank AS score
FROM (SELECT rowid, rank FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?) AS f
JOIN papers p ON p.pk = f.rowid
ORDER BY f.rank
"""


def to_fts_query(text: str) -> str:
    words = []
    for w in _WORD.findall(text.lower()):
        if len(w) > 1 and w not in ENGLISH_STOP_WORDS and w not in words:
            words.append(w)
    return " OR ".join(f'"{w}"' for w in words)


def search(conn: sqlite3.Connection, text: str, k: int = 10) -> list[tuple[str, float]]:
    query = to_fts_query(text)
    if not query:
        return []
    return [(r[0], r[1]) for r in conn.execute(_SQL, (query, k))]
