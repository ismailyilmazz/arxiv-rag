import sqlite3

import numpy as np

from core.search_dense import DenseIndex


def related(conn: sqlite3.Connection, index: DenseIndex, seed_ids: list[str], k: int = 7,
            per_seed: int = 30) -> list[dict]:
    if not seed_ids or k <= 0:
        return []
    rows = conn.execute(f"SELECT id, pk FROM papers WHERE id IN ({','.join('?' * len(seed_ids))})",
                        seed_ids).fetchall()
    positions = index._positions()
    seed_pks = [pk for _, pk in rows if pk < len(positions) and positions[pk] >= 0]
    if not seed_pks:
        return []
    vectors = index.vectors[positions[np.array(seed_pks)]]
    top_pks, top_scores = index.search_batch(vectors, per_seed + len(seed_pks))
    excluded, best = set(seed_pks), {}
    for pks, scores in zip(top_pks, top_scores):
        for pk, score in zip(pks.tolist(), scores.tolist()):
            if pk not in excluded and score > best.get(pk, -2.0):
                best[pk] = score
    chosen = sorted(best.items(), key=lambda item: -item[1])[:k]
    if not chosen:
        return []
    pks = [pk for pk, _ in chosen]
    meta = {r[0]: r[1:] for r in conn.execute(
        f"SELECT pk, id, title, published, abstract FROM papers WHERE pk IN ({','.join('?' * len(pks))})", pks)}
    return [{"id": meta[pk][0], "title": meta[pk][1], "published": meta[pk][2], "abstract": meta[pk][3],
             "score": round(score, 4)} for pk, score in chosen if pk in meta]
