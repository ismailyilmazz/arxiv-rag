import sqlite3
from typing import Callable, Optional

from core import search_bm25

SearchFn = Callable[[sqlite3.Connection, str, int], list[tuple[str, float]]]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def search(conn: sqlite3.Connection, text: str, k: int, dense: SearchFn, depth: int = 100,
           gate: Optional[int] = None) -> list[tuple[str, float]]:
    lexical = [pid for pid, _ in search_bm25.search(conn, text, depth, gate=gate)]
    semantic = [pid for pid, _ in dense(conn, text, depth)]
    return rrf([lexical, semantic])[:k]
