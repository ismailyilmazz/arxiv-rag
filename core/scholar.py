import json
import time
import urllib.error
import urllib.request

from core import config
from core.arxiv_ids import normalize_id

API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "externalIds,citationCount,year"
MIN_INTERVAL = 1.1
_last = [0.0]


def _request(url: str, payload=None, retries: int = 5):
    headers = {"User-Agent": f"arxiv-rag ({config.ARXIV_CONTACT_EMAIL or 'no-contact'})"}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    for attempt in range(retries):
        wait = MIN_INTERVAL - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    return None


def _arxiv(paper: dict):
    raw = ((paper or {}).get("externalIds") or {}).get("ArXiv")
    if not raw:
        return None
    try:
        return normalize_id(raw)
    except ValueError:
        return None


def batch(ids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        data = _request(f"{API}/paper/batch?fields={FIELDS}", {"ids": [f"ARXIV:{pid}" for pid in chunk]}) or []
        for pid, item in zip(chunk, data):
            if item:
                out[pid] = {"citations": int(item.get("citationCount") or 0), "year": item.get("year")}
    return out


def references(pid: str, limit: int = 200) -> list[dict]:
    data = _request(f"{API}/paper/ARXIV:{pid}/references?fields={FIELDS}&limit={limit}") or {}
    out = []
    for row in data.get("data") or []:
        paper = row.get("citedPaper") or {}
        arxiv = _arxiv(paper)
        if arxiv:
            out.append({"id": arxiv, "citations": int(paper.get("citationCount") or 0), "year": paper.get("year")})
    return out
