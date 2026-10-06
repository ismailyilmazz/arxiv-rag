import json
import sqlite3
from datetime import datetime, timezone
from typing import Callable, Optional

from core import fulltext, llm

FIELDS = ("problem", "method", "findings", "limitations")
PROMPT = """You are reading a scientific paper. Using ONLY the text below, write a reading card in English.

Title: {title}

Text (selected sections):
{text}

Return JSON: {{"problem": "...", "method": "...", "findings": "...", "limitations": "..."}}
Each field is 2 to 4 sentences. Keep concrete names, datasets and numbers that appear in the text.
If the text states no limitations, describe what the authors leave open; if nothing is stated, write "Not stated."
"""


def init_cache(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS cards (
        paper_id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT NOT NULL,
        card TEXT NOT NULL, tokens INTEGER NOT NULL, created_at TEXT NOT NULL)""")
    conn.commit()


def is_open_license(license_url) -> bool:
    return bool(license_url) and "creativecommons.org" in license_url


def get_card(cache: sqlite3.Connection, paper: dict, model: str, policy: str = "all",
             fetcher: Optional[Callable[[str], fulltext.FullText]] = None) -> tuple[dict, dict]:
    fetcher = fetcher or fulltext.fetch
    row = cache.execute("SELECT card, source FROM cards WHERE paper_id = ?", (paper["id"],)).fetchone()
    if row is not None:
        return json.loads(row[0]), {"source": row[1], "tokens": 0, "cached": True}

    if policy == "cc" and not is_open_license(paper.get("license")):
        text_source = fulltext.FullText(paper["id"], "abstract", [])
    else:
        text_source = fetcher(paper["id"])
    text = fulltext.select_text(text_source.sections, paper["abstract"])
    data, tokens = llm.chat(PROMPT.format(title=paper["title"], text=text), model=model, max_tokens=900)
    card = {key: " ".join(str(data.get(key) or "").split()) for key in FIELDS}
    if not card["problem"] or not card["method"]:
        raise ValueError(f"{paper['id']}: okuma kartı eksik döndü")

    cache.execute("INSERT OR REPLACE INTO cards VALUES (?, ?, ?, ?, ?, ?)",
                  (paper["id"], text_source.source, model, json.dumps(card, ensure_ascii=False), tokens,
                   datetime.now(timezone.utc).isoformat(timespec="seconds")))
    cache.commit()
    return card, {"source": text_source.source, "tokens": tokens, "cached": False}
