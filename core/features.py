import math
import sqlite3
from typing import Optional

import numpy as np

from core import search_bm25
from core.fields import field_of
from core.lang import is_turkish
from core.search_bm25 import _stem, query_words

RANK_FEATURES = [
    "dense_score", "dense_rank", "bm25_score", "bm25_rank", "in_both",
    "field_prob", "title_overlap", "year", "abstract_len", "query_turkish", "query_len",
]
QUERY_FEATURES = [
    "top1_dense", "dense_gap", "overlap10", "field_maxprob", "field_entropy", "query_turkish", "query_len",
]


class FeatureBuilder:
    def __init__(self, conn: sqlite3.Connection, index, encoder, field_model: Optional[dict] = None,
                 depth: int = 100, gate: int = 2):
        self.conn = conn
        self.index = index
        self.encoder = encoder
        self.field_model = field_model
        self.depth = depth
        self.gate = gate

    def _field_probs(self, vectors: np.ndarray) -> Optional[np.ndarray]:
        if self.field_model is None:
            return None
        return self.field_model["model"].predict_proba(vectors)

    def _metadata(self, pks: list[int]) -> dict[int, tuple]:
        rows = self.conn.execute(
            f"SELECT pk, id, title, length(abstract), primary_category, published FROM papers "
            f"WHERE pk IN ({','.join('?' * len(pks))})", pks)
        return {r[0]: tuple(r[1:]) for r in rows}

    def build(self, texts: list[str], batch_size: int = 64) -> list[dict]:
        vectors = self.encoder.encode_queries(texts, batch_size=batch_size)
        dense_pks, dense_scores = self.index.search_batch(vectors, self.depth)
        probs = self._field_probs(vectors)
        groups = self.field_model["groups"] if self.field_model else []
        group_index = {g: i for i, g in enumerate(groups)}
        missing_rank = self.depth + 1
        results = []
        for i, text in enumerate(texts):
            d_pks = [int(p) for p in dense_pks[i]]
            d_rank = {pk: r for r, pk in enumerate(d_pks, start=1)}
            lexical = search_bm25.search_pks(self.conn, text, self.depth, gate=self.gate)
            b_rank = {pk: r for r, (pk, _) in enumerate(lexical, start=1)}
            b_score = dict(lexical)
            pks = list(dict.fromkeys(d_pks + [pk for pk, _ in lexical]))
            scores = self.index.scores_for(vectors[i], pks)
            meta = self._metadata(pks)
            q_stems = {_stem(w) for w in query_words(text)}
            turkish = float(is_turkish(text))
            q_len = float(len(query_words(text)))
            p = probs[i] if probs is not None else None

            rows, ids = [], []
            for pk, score in zip(pks, scores):
                pid, title, abstract_len, category, published = meta[pk]
                t_stems = {_stem(w) for w in query_words(title)}
                field_prob = float(p[group_index[field_of(category)]]) if p is not None and field_of(category) in group_index else 0.0
                rows.append([
                    float(score), d_rank.get(pk, missing_rank), b_score.get(pk, 0.0), b_rank.get(pk, missing_rank),
                    float(pk in d_rank and pk in b_rank), field_prob,
                    len(q_stems & t_stems) / max(1, len(q_stems)),
                    float(published[:4]) if published else 0.0, float(abstract_len or 0), turkish, q_len,
                ])
                ids.append(pid)

            top10_d = set(d_pks[:10])
            top10_b = {pk for pk, _ in lexical[:10]}
            if p is not None:
                maxprob = float(p.max())
                entropy = float(-sum(x * math.log(x) for x in p if x > 0))
            else:
                maxprob, entropy = 0.0, 0.0
            query_row = [
                float(dense_scores[i][0]), float(dense_scores[i][0] - dense_scores[i][min(9, len(d_pks) - 1)]),
                len(top10_d & top10_b) / 10, maxprob, entropy, turkish, q_len,
            ]
            results.append({"pks": pks, "ids": ids, "X": np.array(rows, dtype=np.float32),
                            "query": np.array(query_row, dtype=np.float32)})
        return results
