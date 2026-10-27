import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Iterator, Optional

from arxiv_rag.storage.arxiv_ids import normalize_id

BASE_URL = "https://oaipmh.arxiv.org/oai"
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arxiv": "http://arxiv.org/OAI/arXiv/"}
DELAY_SECONDS = 3.0


def http_get(url: str, contact: str, retries: int = 8, sleep: Callable = time.sleep) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": f"arxiv-rag/1.0 (mailto:{contact})"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=180) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < retries - 1:
                sleep(int(e.headers.get("Retry-After") or 30))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                sleep(30)
                continue
            raise
    raise RuntimeError("OAI-PMH sunucusuna ulaşılamadı")


def pages(start: str, until: str, contact: str, get: Callable = http_get, sleep: Callable = time.sleep) -> Iterator:
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "from": start, "until": until}
    while True:
        root = ET.fromstring(get(f"{BASE_URL}?{urllib.parse.urlencode(params)}", contact))
        error = root.find("oai:error", NS)
        if error is not None:
            if error.get("code") == "noRecordsMatch":
                return
            raise RuntimeError(f"OAI-PMH hatası {error.get('code')}: {(error.text or '').strip()}")
        yield root
        token = root.find(".//oai:resumptionToken", NS)
        if token is None or not (token.text or "").strip():
            return
        params = {"verb": "ListRecords", "resumptionToken": token.text.strip()}
        sleep(DELAY_SECONDS)


def _text(node, path: str) -> str:
    element = node.find(path, NS)
    return " ".join((element.text or "").split()) if element is not None and element.text else ""


def parse_record(record) -> Optional[dict]:
    header = record.find("oai:header", NS)
    meta = record.find("oai:metadata/arxiv:arXiv", NS)
    if header is None or header.get("status") == "deleted" or meta is None:
        return None
    try:
        pid = normalize_id(_text(meta, "arxiv:id"))
    except ValueError:
        return None
    authors = []
    for author in meta.findall("arxiv:authors/arxiv:author", NS):
        parts = (_text(author, "arxiv:forenames"), _text(author, "arxiv:keyname"), _text(author, "arxiv:suffix"))
        name = " ".join(p for p in parts if p)
        if name:
            authors.append(name)
    categories = _text(meta, "arxiv:categories")
    created = _text(meta, "arxiv:created")
    return {
        "id": pid,
        "title": _text(meta, "arxiv:title"),
        "abstract": _text(meta, "arxiv:abstract"),
        "authors": ", ".join(authors),
        "categories": categories,
        "primary_category": categories.split()[0] if categories else "",
        "published": created,
        "updated": _text(meta, "arxiv:updated") or created,
        "license": _text(meta, "arxiv:license"),
        "doi": _text(meta, "arxiv:doi"),
    }


def harvest(
    start: str, until: str, contact: str, get: Callable = http_get, sleep: Callable = time.sleep
) -> tuple[list[dict], int]:
    rows, skipped = {}, 0
    for root in pages(start, until, contact, get, sleep):
        for record in root.findall("oai:ListRecords/oai:record", NS):
            row = parse_record(record)
            if row is None:
                skipped += 1
                continue
            rows[row["id"]] = row
    return list(rows.values()), skipped


def check(rows: list[dict], limit: int = 200000) -> list[str]:
    problems = []
    if len(rows) > limit:
        problems.append(f"{len(rows)} kayıt makul sınırın ({limit}) üstünde")
    empty = [r["id"] for r in rows if not r["title"] or not r["abstract"]]
    if len(empty) > max(10, len(rows) // 100):
        problems.append(f"{len(empty)} kayıtta başlık ya da özet boş")
    return problems
