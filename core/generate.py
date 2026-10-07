import sqlite3
import time
from typing import Callable, Optional

from core import cards, citations, config, fulltext, writer
from core.arxiv_ids import normalize_id

MAX_SOURCES = 5


def paper_info(conn: sqlite3.Connection, paper_ids: list[str],
               metadata_fetcher: Optional[Callable[[list[str]], dict]] = None) -> dict[str, dict]:
    metadata_fetcher = metadata_fetcher or fulltext.fetch_metadata
    rows = conn.execute(
        f"SELECT id, title, abstract, authors, published, primary_category, license FROM papers "
        f"WHERE id IN ({','.join('?' * len(paper_ids))})", paper_ids).fetchall()
    out = {r[0]: {"title": r[1], "abstract": r[2], "authors": r[3], "published": r[4],
                  "primary_category": r[5], "license": r[6]} for r in rows}
    missing = [p for p in paper_ids if p not in out]
    if missing:
        out.update(metadata_fetcher(missing))
    return out


def generate(conn: sqlite3.Connection, cache: sqlite3.Connection, paper_ids: list[str], doc_type: str = "survey",
             lang: str = "en", card_model: Optional[str] = None, write_model: Optional[str] = None,
             policy: Optional[str] = None, fetcher: Optional[Callable] = None,
             metadata_fetcher: Optional[Callable] = None) -> dict:
    fetcher = fetcher or fulltext.fetch
    if doc_type not in writer.DOC_TYPES:
        raise ValueError(f"Bilinmeyen tür: {doc_type}. Seçenekler: {', '.join(writer.DOC_TYPES)}")
    if lang not in writer.LANGUAGES:
        raise ValueError(f"Bilinmeyen dil: {lang}. Seçenekler: {', '.join(writer.LANGUAGES)}")
    ids = list(dict.fromkeys(normalize_id(p) for p in paper_ids))
    if not 1 <= len(ids) <= MAX_SOURCES:
        raise ValueError(f"1 ile {MAX_SOURCES} arasında makale seçilmeli.")

    card_model = card_model or config.LLM_CARD_MODEL
    write_model = write_model or config.LLM_WRITE_MODEL
    policy = policy or config.FULLTEXT_POLICY
    cards.init_cache(cache)

    start = time.perf_counter()
    info = paper_info(conn, ids, metadata_fetcher)
    missing = [p for p in ids if p not in info]
    if missing:
        raise ValueError(f"Makale bilgisi bulunamadı: {', '.join(missing)}")

    sources, card_tokens = [], 0
    for n, pid in enumerate(ids, start=1):
        paper = {"id": pid, **info[pid]}
        card, meta = cards.get_card(cache, paper, card_model, policy=policy, fetcher=fetcher)
        text = cards.get_text(cache, paper, policy, fetcher)
        card_tokens += meta["tokens"]
        sources.append({"n": n, "id": pid, **info[pid], "card": card, "excerpt": fulltext.excerpt(text.sections),
                        "text_source": meta["source"], "cached": meta["cached"]})
    t_cards = time.perf_counter()

    draft, write_tokens, sections = writer.write_document(sources, doc_type, lang, write_model)
    t_write = time.perf_counter()
    body, report = citations.check_and_fix(draft, len(sources))
    markdown = body.rstrip() + "\n\n" + citations.bibliography(sources, lang)

    return {
        "markdown": markdown,
        "doc_type": doc_type,
        "lang": lang,
        "models": {"card": card_model, "write": write_model},
        "sources": [{"n": s["n"], "id": s["id"], "title": s["title"], "text_source": s["text_source"],
                     "cached": s["cached"]} for s in sources],
        "citations": report,
        "length": {"words": len(body.split()), "truncated": any(r["truncated"] for r in sections)},
        "sections": sections,
        "tokens": {"cards": card_tokens, "write": write_tokens, "total": card_tokens + write_tokens},
        "timings_ms": {"cards": round((t_cards - start) * 1000), "write": round((t_write - t_cards) * 1000),
                       "total": round((time.perf_counter() - start) * 1000)},
    }
