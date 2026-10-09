import math
import re
import sqlite3
import time
from collections import Counter
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from core import cards, citations, config, coverage, fulltext, llm, scholar, writer
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


_SURVEY = re.compile(r"\b(survey|review|overview|tutorial|primer|state of the art|systematic literature)\b",
                     re.IGNORECASE)


def is_survey(title: str) -> bool:
    return bool(_SURVEY.search(title or ""))


def citation_scores(info: dict[str, dict], now_year: int) -> dict[str, float]:
    per_year = {pid: d["citations"] / max(1, now_year - (d.get("year") or now_year) + 1) for pid, d in info.items()}
    top = max((math.log1p(v) for v in per_year.values()), default=0.0)
    return {pid: (math.log1p(v) / top if top else 0.0) for pid, v in per_year.items()}


def rerank(fused: dict[str, float], cite: dict[str, float], weight: float = 0.4) -> dict[str, float]:
    top = max(fused.values(), default=0.0) or 1.0
    return {pid: (1 - weight) * f / top + weight * cite.get(pid, 0.0) for pid, f in fused.items()}


def add_references(fused: dict[str, float], refs_by_anchor: list[list[dict]], in_corpus: set, exclude: set,
                   per_anchor: int = 30) -> int:
    added = 0
    for refs in refs_by_anchor:
        ranked = sorted((r for r in refs if r["id"] in in_corpus and r["id"] not in exclude),
                        key=lambda r: -r["citations"])[:per_anchor]
        for rank, r in enumerate(ranked, start=1):
            added += r["id"] not in fused
            fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (60 + rank)
    return added


def screen(scores: dict[str, float], seeds: list[str], n_sources: int, titles: dict[str, str] = None,
           max_surveys: int = 2) -> tuple[list[str], list[str]]:
    titles = titles or {}
    picked, moved, surveys = [], [], 0
    for pid, _ in sorted(scores.items(), key=lambda kv: -kv[1]):
        if pid in seeds:
            continue
        if len(seeds) + len(picked) >= n_sources:
            break
        if is_survey(titles.get(pid, "")):
            if surveys >= max_surveys:
                moved.append(pid)
                continue
            surveys += 1
        picked.append(pid)
    return list(seeds) + picked, moved[:5]


def merge_small(labels: np.ndarray, vectors: np.ndarray, min_size: int = 3) -> np.ndarray:
    labels = np.asarray(labels).copy()
    while True:
        counts = Counter(labels.tolist())
        small = [c for c, n in counts.items() if n < min_size]
        if not small or len(counts) <= 1:
            break
        c = min(small, key=lambda k: counts[k])
        own = vectors[labels == c].mean(axis=0)
        others = {k: vectors[labels == k].mean(axis=0) for k in counts if k != c}
        target = max(others, key=lambda k: float(own @ others[k]) /
                     (float(np.linalg.norm(own) * np.linalg.norm(others[k])) + 1e-9))
        labels[labels == c] = target
    mapping = {c: i for i, c in enumerate(sorted(set(labels.tolist())))}
    return np.array([mapping[c] for c in labels.tolist()], dtype=int)


def topical_relevance(conn: sqlite3.Connection, index, fused: dict[str, float], seeds: list[str], extra=(),
                      top: int = 10, pct: float = 25, margin: float = 0.05) -> tuple[dict[str, float], float]:
    searched = [pid for pid, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    pool = list(dict.fromkeys(searched + sorted(extra)))
    if not pool:
        return {}, -1.0
    ids, vectors = vectors_for(conn, index, pool)
    row = {pid: i for i, pid in enumerate(ids)}
    anchor_ids = [p for p in seeds if p in row] + [p for p in searched[:top] if p in row]
    if not anchor_ids:
        return {}, -1.0
    anchor = vectors[[row[p] for p in anchor_ids]].mean(axis=0)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)
    norms = np.linalg.norm(vectors, axis=1) + 1e-9
    relevance = {pid: float(vectors[i] @ anchor / norms[i]) for pid, i in row.items()}
    base = [relevance[p] for p in searched if p in relevance]
    return relevance, float(np.percentile(base, pct)) - margin


def material_text(s: dict) -> str:
    card = " ".join(str(v) for v in (s.get("card") or {}).values())
    return " ".join([s.get("title") or "", s.get("abstract") or "", s.get("key_sentences") or "", card])


def vectors_for(conn: sqlite3.Connection, index, ids: list[str]) -> tuple[list[str], np.ndarray]:
    pk_of = dict(conn.execute(f"SELECT id, pk FROM papers WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall())
    positions = index._positions()
    kept = [pid for pid in ids if pid in pk_of and pk_of[pid] < len(positions) and positions[pk_of[pid]] >= 0]
    rows = positions[np.array([pk_of[pid] for pid in kept], dtype=np.int64)] if kept else np.array([], dtype=np.int64)
    return kept, np.asarray(index.vectors[rows], dtype=np.float32)


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


def pick_deep(themes: list[dict], ids: list[str], vectors: np.ndarray, seeds: list[str], max_deep: int,
              titles: dict[str, str] = None) -> list[str]:
    titles = titles or {}
    deep = [s for s in seeds if s in ids][:max_deep]
    row = {pid: i for i, pid in enumerate(ids)}
    for theme in themes:
        if len(deep) >= max_deep:
            break
        if any(pid in deep for pid in theme["ids"]):
            continue
        members = [pid for pid in theme["ids"] if pid in row and not is_survey(titles.get(pid, ""))]
        if not members:
            continue
        centroid = vectors[[row[pid] for pid in members]].mean(axis=0)
        sims = [float(vectors[row[pid]] @ centroid) for pid in members]
        deep.append(members[int(np.argmax(sims))])
    return deep


def build_broad(conn: sqlite3.Connection, cache: sqlite3.Connection, pipeline, topic: str,
                doc_type: str = "survey", seed_ids: list[str] = (), lang: str = "en", n_sources: int = 30,
                max_deep: int = 6, exclude: tuple = (), card_model: Optional[str] = None,
                write_model: Optional[str] = None, fetcher: Optional[Callable] = None, policy: Optional[str] = None,
                focus: Optional[str] = None, use_scholar: bool = True) -> dict:
    if doc_type not in writer.BROAD:
        raise ValueError(f"Bilinmeyen tür: {doc_type}. Seçenekler: {', '.join(writer.BROAD)}")
    card_model = card_model or config.LLM_CARD_MODEL
    write_model = write_model or config.LLM_WRITE_MODEL
    policy = policy or config.FULLTEXT_POLICY
    seeds = list(dict.fromkeys(normalize_id(s) for s in seed_ids))
    excluded = {normalize_id(e) for e in exclude}
    subject = f"{topic} (focus: {focus})" if focus else topic
    cards.init_cache(cache)
    tokens, timings, start = {}, {}, time.perf_counter()

    queries, tokens["expand"] = expand_queries(subject, write_model)
    near = [r["id"] for r in related(conn, pipeline.index, seeds, k=10)] if seeds else []
    fused, rejected = collect(pipeline, queries, near, exclude=excluded)
    s2 = {"ok": False, "anchors": [], "reference_additions": 0, "references_rejected": 0, "with_citations": 0,
          "relevance_threshold": None}
    cite = {}
    if use_scholar and fused:
        try:
            anchors = seeds or [pid for pid, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:3]]
            refs = [scholar.references(pid) for pid in anchors]
            in_corpus = coverage.in_corpus(conn, {r["id"] for rs in refs for r in rs}) - excluded
            relevance, threshold = topical_relevance(conn, pipeline.index, fused, seeds, in_corpus)
            relevant = lambda pid: relevance.get(pid, -2.0) >= threshold
            allowed = {pid for pid in in_corpus if relevant(pid)}
            s2["references_rejected"] = len({pid for pid in in_corpus - allowed if pid not in fused})
            s2["reference_additions"] = add_references(fused, refs, allowed, excluded | set(seeds))
            info = scholar.batch(list(fused))
            cite = {pid: v for pid, v in citation_scores(info, datetime.now().year).items() if relevant(pid)}
            s2.update(ok=True, anchors=anchors, with_citations=len(info), relevance_threshold=round(threshold, 4))
        except Exception as e:
            s2["error"] = f"{type(e).__name__}: {e}"[:200]
    scores = rerank(fused, cite)
    pool = list(fused)
    titles = {r[0]: r[1] for i in range(0, len(pool), 900) for r in conn.execute(
        f"SELECT id, title FROM papers WHERE id IN ({','.join('?' * len(pool[i:i + 900]))})", pool[i:i + 900])}
    screened, moved = screen(scores, seeds, n_sources, titles)
    ids, vectors = vectors_for(conn, pipeline.index, screened)
    if not ids:
        raise ValueError("Bu konu için yeterli makale bulunamadı (sorgular reddedildi ya da sonuç çıkmadı).")
    labels = merge_small(cluster(vectors), vectors)
    wanted = ids + [m for m in moved if m not in ids]
    meta = {r[0]: {"title": r[1], "abstract": r[2], "authors": r[3], "published": r[4], "license": r[5]}
            for r in conn.execute(f"SELECT id, title, abstract, authors, published, license FROM papers "
                                  f"WHERE id IN ({','.join('?' * len(wanted))})", wanted)}
    themes, tokens["themes"] = name_themes(subject, ids, labels, {pid: meta[pid]["title"] for pid in ids}, write_model)
    timings["retrieval_ms"] = round((time.perf_counter() - start) * 1000)

    kept = [pid for t in themes for pid in t["ids"]]
    deep = pick_deep(themes, ids, vectors, seeds, max_deep, titles)
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
    draft, tokens["write"], sections = writer.write_broad(doc_type, sources, themes, subject, lang, write_model)
    timings["write_ms"] = round((time.perf_counter() - t_write) * 1000)

    body, report = citations.check_and_fix(draft, len(sources), tuple(writer.broad_uncited_allowed(doc_type, lang)))
    body, mapping = citations.renumber(body)
    for s in sources:
        s["n_old"], s["n"] = s["n"], mapping.get(s["n"])
    cited = sorted([s for s in sources if s["n"]], key=lambda s: s["n"])
    further = [s for s in sources if not s["n"]] + [{"id": m, **meta[m]} for m in moved if m in meta]
    report["name_mismatches"] = citations.name_mismatches(body, cited)
    report["meta_language"] = citations.meta_language(body)
    report["number_audit"] = citations.number_audit(body, {s["n"]: material_text(s) for s in cited})
    markdown = body.rstrip() + "\n\n" + citations.bibliography(cited, lang) + "\n" + \
        citations.further_reading(further, lang)
    tokens["total"] = sum(tokens.values())
    timings["total_ms"] = round((time.perf_counter() - start) * 1000)
    return {
        "markdown": markdown, "topic": topic, "focus": focus, "doc_type": doc_type, "lang": lang,
        "queries": queries, "rejected_queries": rejected, "scholar": s2,
        "candidates": len(fused), "screened": len(screened), "surveys_moved": moved,
        "candidate_ids": list(fused), "screened_ids": screened,
        "themes": [{"name": t["name"], "size": len(t["ids"]),
                    "deep": [pid for pid in t["ids"] if pid in deep]} for t in themes],
        "deep": deep, "cited": [s["id"] for s in cited], "further_reading": [s["id"] for s in further],
        "citations": report, "sections": sections, "tokens": tokens, "timings_ms": timings,
        "length": {"words": len(body.split()), "truncated": any(r["truncated"] for r in sections)},
    }
