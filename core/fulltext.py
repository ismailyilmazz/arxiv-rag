import io
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from pypdf import PdfReader

from core import config
from core.arxiv_ids import normalize_id

MIN_INTERVAL = 3.0
SECTION_KEYS = {
    "introduction": ("introduction",),
    "method": ("method", "approach", "model", "framework", "proposed", "architecture"),
    "results": ("experiment", "result", "evaluation"),
    "conclusion": ("conclusion", "discussion", "limitation", "summary"),
}
BUDGET = {"introduction": 3500, "method": 5000, "results": 3500, "conclusion": 2000}
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_PDF_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+([A-Z][A-Za-z ,&\-]{2,70})\s*$")
_PDF_PLAIN_HEADING = re.compile(r"^\s*(Abstract|Introduction|Conclusions?|Discussion)\s*$", re.IGNORECASE)
_PDF_REFERENCES = re.compile(r"^\s*(References|Bibliography)\s*$", re.IGNORECASE)
_last_request = [0.0]


@dataclass
class FullText:
    paper_id: str
    source: str
    sections: list[tuple[str, str]] = field(default_factory=list)


def _get(url: str) -> tuple[int, bytes]:
    wait = _last_request[0] + MIN_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    contact = f" (mailto:{config.ARXIV_CONTACT_EMAIL})" if config.ARXIV_CONTACT_EMAIL else ""
    request = urllib.request.Request(url, headers={"User-Agent": f"arxiv-rag/0.1{contact}"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except urllib.error.URLError:
        return 0, b""
    finally:
        _last_request[0] = time.monotonic()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_html(html: bytes) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="ltx_document")
    if article is None:
        return []
    for tag in article.select(".ltx_bibliography, .ltx_appendix, figure, table, .ltx_note, .ltx_authors"):
        tag.decompose()
    for math in article.find_all("math"):
        math.replace_with(f" {math.get('alttext', '')} ")
    sections = []
    abstract = article.find(class_="ltx_abstract")
    if abstract is not None:
        heading = abstract.find(class_="ltx_title")
        if heading is not None:
            heading.decompose()
        sections.append(("Abstract", _clean(abstract.get_text(" "))))
        abstract.decompose()
    for section in article.find_all("section", class_="ltx_section"):
        heading = section.find(class_="ltx_title")
        title = _clean(heading.get_text(" ")) if heading is not None else ""
        if heading is not None:
            heading.decompose()
        text = _clean(section.get_text(" "))
        if text:
            sections.append((title, text))
    return sections


def parse_pdf(data: bytes) -> list[tuple[str, str]]:
    reader = PdfReader(io.BytesIO(data))
    lines = "\n".join(page.extract_text() or "" for page in reader.pages).splitlines()
    sections, title, buffer = [], "Body", []

    def flush():
        text = _clean(re.sub(r"(\w)-\n(\w)", r"\1\2", "\n".join(buffer)))
        if text:
            sections.append((title, text))

    for line in lines:
        if _PDF_REFERENCES.match(line):
            break
        if (_PDF_HEADING.match(line) or _PDF_PLAIN_HEADING.match(line)) and len(line.split()) <= 8:
            flush()
            title, buffer = line.strip(), []
        else:
            buffer.append(line)
    flush()
    return sections


def fetch(paper_id: str) -> FullText:
    status, body = _get(f"https://arxiv.org/html/{paper_id}")
    if status == 200:
        sections = parse_html(body)
        if sections:
            return FullText(paper_id, "html", sections)
    status, body = _get(f"https://arxiv.org/pdf/{paper_id}")
    if status == 200 and body[:4] == b"%PDF":
        try:
            sections = parse_pdf(body)
        except Exception:
            sections = []
        if sections:
            return FullText(paper_id, "pdf", sections)
    return FullText(paper_id, "abstract", [])


def select_text(sections: list[tuple[str, str]], abstract: str) -> str:
    parts = [f"Abstract: {_clean(abstract)}"]
    used = set()
    for key, words in SECTION_KEYS.items():
        for i, (title, text) in enumerate(sections):
            if i not in used and title != "Abstract" and any(w in title.lower() for w in words):
                parts.append(f"{title}: {text[:BUDGET[key]]}")
                used.add(i)
                break
    if len(parts) == 1:
        body = " ".join(text for title, text in sections if title != "Abstract")[:sum(BUDGET.values())]
        if body:
            parts.append(f"Body: {body}")
    return "\n\n".join(parts)


def fetch_metadata(paper_ids: list[str]) -> dict[str, dict]:
    if not paper_ids:
        return {}
    status, body = _get("https://export.arxiv.org/api/query?id_list=" + ",".join(paper_ids)
                        + f"&max_results={len(paper_ids)}")
    if status != 200:
        return {}
    out = {}
    for entry in ET.fromstring(body).findall("a:entry", _ATOM):
        try:
            pid = normalize_id(entry.find("a:id", _ATOM).text)
        except (ValueError, AttributeError):
            continue
        category = entry.find("arxiv:primary_category", _ATOM)
        out[pid] = {
            "title": _clean(entry.find("a:title", _ATOM).text),
            "abstract": _clean(entry.find("a:summary", _ATOM).text),
            "authors": ", ".join(_clean(a.find("a:name", _ATOM).text) for a in entry.findall("a:author", _ATOM)),
            "published": entry.find("a:published", _ATOM).text[:10],
            "primary_category": category.get("term") if category is not None else None,
            "license": None,
        }
    return out
