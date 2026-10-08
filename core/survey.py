import re
import sqlite3
import time
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from core import cards, citations, config, fulltext, llm, writer
from core.arxiv_ids import normalize_id
from core.related import related

EXPAND_PROMPT = """Topic: {topic}

Write 5 short English search queries (3 to 8 words each) that together cover the subtopics a literature survey on this topic must include: core methods, important variants, training and data, systems and efficiency, evaluation and limitations.
Return JSON: {{"queries": ["...", "..."]}}"""

THEME_PROMPT = """Topic: {topic}

These are clusters of paper titles collected for a literature survey on this topic:
{clusters}

For every cluster give a short theme name (2 to 6 words) and one sentence that describes what unites its papers. Mark a cluster as off_topic when most of its papers are unrelated to the topic. Also give the order in which the themes should be presented in the survey.
Return JSON: {{"themes": [{{"cluster": 0, "name": "...", "description": "...", "off_topic": false}}], "order": [0, 1]}}"""

_CUE = re.compile(r"\b(we|our|this paper|this work|propose|introduce|achiev|outperform)\w*|\d", re.IGNORECASE)


def key_sentences(abstract: str, n: int = 3, max_chars: int = 480) -> str:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", " ".join((abstract or "").split())) if s]
    picked = [s for s in sentences if _CUE.search(s)] or sentences
    return " ".join(picked[:n])[:max_chars]


def expand_queries(topic: str, model: str) -> tuple[list[str], int]:
    data, tokens = llm.chat(EXPAND_PROMPT.format(topic=topic), model=model, max_tokens=800)
    queries = [" ".join(str(q).split()) for q in data.get("queries", []) if str(q).strip()]
    return list(dict.fromkeys([topic] + queries[:6])), tokens


def collect(pipeline, queries: list[str], extra: list[str] = (), per_query: int = 20,
            exclude: set = frozenset()) -> tuple[dict[str, float], list[str]]:
    fused, rejected = {}, []
    for query in queries:
        out = pipeline.search(query, k=per_query)
        if not out["accepted"]:
            rejected.append(query)
            continue
        for rank, r in enumerate(out["results"], start=1):
            if r["id"] not in exclude:
                fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (60 + rank)
    for rank, pid in enumerate(extra, start=1):
        if pid not in exclude:
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (60 + rank)
    return fused, rejected


def screen(fused: dict[str, float], seeds: list[str], n_sources: int) -> list[str]:
    ranked = [pid for pid, _ in sorted(fused.items(), key=lambda kv: -kv[1]) if pid not in seeds]
    return list(seeds) + ranked[:max(0, n_sources - len(seeds))]


def vectors_for(conn: sqlite3.Connection, index, ids: list[str]) -> tuple[list[str], np.ndarray]:
    pk_of = dict(conn.execute(f"SELECT id, pk FROM papers WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall())
    positions = index._positions()
    kept = [pid for pid in ids if pid in pk_of and pk_of[pid] < len(positions) and positions[pk_of[pid]] >= 0]
    rows = positions[np.array([pk_of[pid] for pid in kept], dtype=np.int64)] if kept else np.array([], dtype=np.int64)
    return kept, index.vectors[rows]


def cluster(vectors: np.ndarray, k_range: tuple[int, int] = (3, 6), seed: int = 42) -> np.ndarray:
    n = len(vectors)
    if n < 4:
        return np.zeros(n, dtype=int)
    best, best_score = None, -2.0
    for k in range(k_range[0], min(k_range[1], n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)
        score = silhouette_score(vectors, labels, metric="cosine")
        if score > best_score:
            best, best_score = labels, score
    return best if best is not None else np.zeros(n, dtype=int)


def name_themes(topic: str, ids: list[str], labels: np.ndarray, titles: dict[str, str],
                model: str) -> tuple[list[dict], int]:
    clusters = "\n".join(
        f"Cluster {c}:\n" + "\n".join([f"- {titles[pid]}" for pid, lab in zip(ids, labels) if lab == c][:12])
        for c in sorted(set(labels.tolist())))
    data, tokens = llm.chat(THEME_PROMPT.format(topic=topic, clusters=clusters), model=model, max_tokens=1200)
    present = sorted(set(labels.tolist()))
    named = {int(t["cluster"]): t for t in data.get("themes", []) if str(t.get("cluster", "")).lstrip("-").isdigit()}
    raw = [int(c) for c in data.get("order", []) if str(c).lstrip("-").isdigit() and int(c) in present]
    order = raw + [c for c in present if c not in raw]
    themes = []
    for c in dict.fromkeys(order):
        t = named.get(c, {})
        if t.get("off_topic"):
            continue
        themes.append({"cluster": c, "name": str(t.get("name") or f"Theme {c + 1}").strip(),
                       "description": str(t.get("description") or "").strip(),
                       "ids": [pid for pid, lab in zip(ids, labels) if lab == c]})
    return themes, tokens


def pick_deep(themes: list[dict], ids: list[str], vectors: np.ndarray, seeds: list[str], max_deep: int) -> list[str]:
    deep = [s for s in seeds if s in ids][:max_deep]
    row = {pid: i for i, pid in enumerate(ids)}
    for theme in themes:
        if len(deep) >= max_deep:
            break
        if any(pid in deep for pid in theme["ids"]):
            continue
        members = [pid for pid in theme["ids"] if pid in row]
        if not members:
            continue
        centroid = vectors[[row[pid] for pid in members]].mean(axis=0)
        sims = [float(vectors[row[pid]] @ centroid) for pid in members]
        deep.append(members[int(np.argmax(sims))])
    return deep


def build_broad(conn: sqlite3.Connection, cache: sqlite3.Connection, pipeline, topic: str,
                doc_type: str = "survey", seed_ids: list[str] = (), lang: str = "en", n_sources: int = 30,
                max_deep: int = 6,
                exclude: tuple = (), card_model: Optional[str] = None, write_model: Optional[str] = None,
                fetcher: Optional[Callable] = None, policy: Optional[str] = None) -> dict:
    if doc_type not in writer.BROAD:
        raise ValueError(f"Bilinmeyen tür: {doc_type}. Seçenekler: {', '.join(writer.BROAD)}")
    card_model = card_model or config.LLM_CARD_MODEL
    write_model = write_model or config.LLM_WRITE_MODEL
    policy = policy or config.FULLTEXT_POLICY
    seeds = list(dict.fromkeys(normalize_id(s) for s in seed_ids))
    excluded = {normalize_id(e) for e in exclude}
    cards.init_cache(cache)
    tokens, timings, start = {}, {}, time.perf_counter()

    queries, tokens["expand"] = expand_queries(topic, write_model)
    near = [r["id"] for r in related(conn, pipeline.index, seeds, k=10)] if seeds else []
    fused, rejected = collect(pipeline, queries, near, exclude=excluded)
    screened = screen(fused, seeds, n_sources)
    ids, vectors = vectors_for(conn, pipeline.index, screened)
    if not ids:
        raise ValueError("Bu konu için yeterli makale bulunamadı (sorgular reddedildi ya da sonuç çıkmadı).")
    labels = cluster(vectors)
    meta = {r[0]: {"title": r[1], "abstract": r[2], "authors": r[3], "published": r[4], "license": r[5]}
            for r in conn.execute(f"SELECT id, title, abstract, authors, published, license FROM papers "
                                  f"WHERE id IN ({','.join('?' * len(ids))})", ids)}
    themes, tokens["themes"] = name_themes(topic, ids, labels, {pid: meta[pid]["title"] for pid in ids}, write_model)
    timings["retrieval_ms"] = round((time.perf_counter() - start) * 1000)

    kept = [pid for t in themes for pid in t["ids"]]
    deep = pick_deep(themes, ids, vectors, seeds, max_deep)
    sources, number = [], {}
    for pid in deep + [p for p in kept if p not in deep]:
        number[pid] = len(sources) + 1
        sources.append({"n": number[pid], "id": pid, **meta[pid],
                        "theme": next((t["name"] for t in themes if pid in t["ids"]), "")})
    tokens["cards"], t_cards = 0, time.perf_counter()
    for s in sources:
        if s["id"] in deep:
            card, info = cards.get_card(cache, {"id": s["id"], **meta[s["id"]]}, card_model, policy=policy,
                                        fetcher=fetcher)
            s["card"] = card
            tokens["cards"] += info["tokens"]
        else:
            s["key_sentences"] = key_sentences(meta[s["id"]]["abstract"])
    timings["cards_ms"] = round((time.perf_counter() - t_cards) * 1000)

    t_write = time.perf_counter()
    draft, tokens["write"], sections = writer.write_broad(doc_type, sources, themes, topic, lang, write_model)
    timings["write_ms"] = round((time.perf_counter() - t_write) * 1000)

    body, report = citations.check_and_fix(draft, len(sources), tuple(writer.broad_uncited_allowed(doc_type, lang)))
    body, mapping = citations.renumber(body)
    for s in sources:
        s["n_old"], s["n"] = s["n"], mapping.get(s["n"])
    cited = sorted([s for s in sources if s["n"]], key=lambda s: s["n"])
    further = [s for s in sources if not s["n"]]
    report["name_mismatches"] = citations.name_mismatches(body, cited)
    report["meta_language"] = citations.meta_language(body)
    markdown = body.rstrip() + "\n\n" + citations.bibliography(cited, lang) + "\n" + \
        citations.further_reading(further, lang)
    tokens["total"] = sum(tokens.values())
    timings["total_ms"] = round((time.perf_counter() - start) * 1000)
    return {
        "markdown": markdown, "topic": topic, "doc_type": doc_type, "lang": lang, "queries": queries, "rejected_queries": rejected,
        "candidates": len(fused), "screened": len(screened),
        "candidate_ids": list(fused), "screened_ids": screened,
        "themes": [{"name": t["name"], "size": len(t["ids"]),
                    "deep": [pid for pid in t["ids"] if pid in deep]} for t in themes],
        "deep": deep, "cited": [s["id"] for s in cited], "further_reading": [s["id"] for s in further],
        "citations": report, "sections": sections, "tokens": tokens, "timings_ms": timings,
        "length": {"words": len(body.split()), "truncated": any(r["truncated"] for r in sections)},
    }
