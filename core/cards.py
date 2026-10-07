import json
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Optional

from core import config, fulltext, llm

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
    conn.execute("""CREATE TABLE IF NOT EXISTS texts (
        paper_id TEXT PRIMARY KEY, source TEXT NOT NULL, sections TEXT NOT NULL, created_at TEXT NOT NULL)""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(cards)")}
    if "version" not in columns:
        conn.execute("ALTER TABLE cards ADD COLUMN version INTEGER NOT NULL DEFAULT 1")
    conn.commit()


def is_open_license(license_url) -> bool:
    return bool(license_url) and "creativecommons.org" in license_url


def is_bad_json(error: Exception) -> bool:
    return isinstance(error, json.JSONDecodeError) or "json_validate_failed" in str(error)


def get_text(cache: sqlite3.Connection, paper: dict, policy: str = "all",
             fetcher: Optional[Callable[[str], fulltext.FullText]] = None) -> fulltext.FullText:
    if policy == "cc" and not is_open_license(paper.get("license")):
        return fulltext.FullText(paper["id"], "abstract", [])
    row = cache.execute("SELECT source, sections FROM texts WHERE paper_id = ?", (paper["id"],)).fetchone()
    if row is not None:
        return fulltext.FullText(paper["id"], row[0], [tuple(x) for x in json.loads(row[1])])
    text = (fetcher or fulltext.fetch)(paper["id"])
    cache.execute("INSERT OR REPLACE INTO texts VALUES (?, ?, ?, ?)",
                  (paper["id"], text.source, json.dumps(text.sections, ensure_ascii=False),
                   datetime.now(timezone.utc).isoformat(timespec="seconds")))
    cache.commit()
    return text


def get_card(cache: sqlite3.Connection, paper: dict, model: str, policy: str = "all",
             fetcher: Optional[Callable[[str], fulltext.FullText]] = None) -> tuple[dict, dict]:
    row = cache.execute("SELECT card, source FROM cards WHERE paper_id = ? AND version = ?",
                        (paper["id"], VERSION)).fetchone()
    if row is not None:
        return json.loads(row[0]), {"source": row[1], "tokens": 0, "cached": True}

    text_source = get_text(cache, paper, policy, fetcher)
    text = fulltext.select_text(text_source.sections, paper["abstract"])
    prompt = PROMPT.format(title=paper["title"], text=text)
    card, tokens, last_error = None, 0, None
    for attempt_model in (model, model, config.LLM_WRITE_MODEL):
        try:
            data, tokens = llm.chat(prompt, model=attempt_model, max_tokens=2000)
        except Exception as e:
            if not is_bad_json(e):
                raise
            last_error = e
            continue
        candidate = {key: " ".join(str(data.get(key) or "").split()) for key in FIELDS}
        if candidate["problem"] and candidate["method"]:
            card, model = candidate, attempt_model
            break
        last_error = ValueError("eksik alanlar")
    if card is None:
        raise ValueError(f"{paper['id']}: okuma kartı üretilemedi ({type(last_error).__name__})")

    cache.execute("INSERT OR REPLACE INTO cards (paper_id, source, model, card, tokens, created_at, version) "
                  "VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (paper["id"], text_source.source, model, json.dumps(card, ensure_ascii=False), tokens,
                   datetime.now(timezone.utc).isoformat(timespec="seconds"), VERSION))
    cache.commit()
    return card, {"source": text_source.source, "tokens": tokens, "cached": False}
