import json
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Optional

from core import fulltext, llm

VERSION = 3
FIELDS = ("problem", "method", "setup", "findings", "limitations")
PROMPT = """You are reading a scientific paper. Using ONLY the text below, write a detailed reading card in English.

Title: {title}

Text (selected sections):
{text}

Return JSON: {{"problem": "...", "method": "...", "setup": "...", "findings": "...", "limitations": "..."}}
- problem: what problem is addressed and why it matters.
- method: how the approach works, with its key components and design choices.
- setup: data sets, models, baselines, scale and evaluation measures that are used.
- findings: main results; copy every concrete number, comparison and claim exactly as stated.
- limitations: limitations or open questions the authors state; if none are stated, write "Not stated."
Each field has 3 to 8 sentences: as many as the text supports, no more. Use only information from the text; do not add outside knowledge.
"""


def init_cache(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS cards (
        paper_id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT NOT NULL,
        card TEXT NOT NULL, tokens INTEGER NOT NULL, created_at TEXT NOT NULL)""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(cards)")}
    if "version" not in columns:
        conn.execute("ALTER TABLE cards ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
    conn.commit()


def is_open_license(license_url) -> bool:
    return bool(license_url) and "creativecommons.org" in license_url


def get_card(cache: sqlite3.Connection, paper: dict, model: str, policy: str = "all",
             fetcher: Optional[Callable[[str], fulltext.FullText]] = None) -> tuple[dict, dict]:
    fetcher = fetcher or fulltext.fetch
    row = cache.execute("SELECT card, source FROM cards WHERE paper_id = ? AND version = ?",
                        (paper["id"], VERSION)).fetchone()
    if row is not None:
        return json.loads(row[0]), {"source": row[1], "tokens": 0, "cached": True}

    if policy == "cc" and not is_open_license(paper.get("license")):
        text_source = fulltext.FullText(paper["id"], "abstract", [])
    else:
        text_source = fetcher(paper["id"])
    text = fulltext.select_text(text_source.sections, paper["abstract"])
    data, tokens = llm.chat(PROMPT.format(title=paper["title"], text=text), model=model, max_tokens=2000)
    card = {key: " ".join(str(data.get(key) or "").split()) for key in FIELDS}
    if not card["problem"] or not card["method"]:
        raise ValueError(f"{paper['id']}: okuma kartı eksik döndü")

    cache.execute("INSERT OR REPLACE INTO cards (paper_id, source, model, card, tokens, created_at, version) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (paper["id"], text_source.source, model, json.dumps(card, ensure_ascii=False), tokens,
                   datetime.now(timezone.utc).isoformat(timespec="seconds"), VERSION))
    cache.commit()
    return card, {"source": text_source.source, "tokens": tokens, "cached": False}
