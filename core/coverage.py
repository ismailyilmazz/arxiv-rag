import re
import sqlite3

from core import fulltext
from core.arxiv_ids import normalize_id

_ARXIV = re.compile(r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:\s?)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?",
                    re.IGNORECASE)


def ids_in_text(text: str) -> set[str]:
    out = set()
    for raw in _ARXIV.findall(text):
        try:
            out.add(normalize_id(raw))
        except ValueError:
            continue
    return out


def fetch_gold(survey_id: str) -> set[str]:
    status, body = fulltext._get(f"https://arxiv.org/html/{survey_id}")
    if status != 200:
        return set()
    return ids_in_text(body.decode("utf-8", errors="ignore")) - {normalize_id(survey_id)}


def in_corpus(conn: sqlite3.Connection, ids: set[str]) -> set[str]:
    ids = sorted(ids)
    if not ids:
        return set()
    return {r[0] for r in conn.execute(f"SELECT id FROM papers WHERE id IN ({','.join('?' * len(ids))})", ids)}


def coverage(gold: set[str], candidates, screened, cited) -> dict:
    n = len(gold)
    rate = lambda found: round(len(gold & set(found)) / n, 4) if n else None
    return {"gold_in_corpus": n, "candidates": rate(candidates), "screened": rate(screened), "cited": rate(cited),
            "cited_gold": sorted(gold & set(cited))}
