import sqlite3
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from core import embeddings
from core.features import FeatureBuilder
from core.search_dense import DenseIndex


class SearchPipeline:
    def __init__(self, conn: sqlite3.Connection, index: DenseIndex, encoder, field_model: dict,
                 ranker: lgb.Booster, guard: dict, depth: int = 100, gate: int = 2):
        self.index = index
        self.ranker = ranker
        self.guard = guard
        self.builder = FeatureBuilder(conn, index, encoder, field_model, depth=depth, gate=gate)

    @classmethod
    def load(cls, conn: sqlite3.Connection, vectors_dir: Path, models_dir: Path) -> "SearchPipeline":
        models_dir = Path(models_dir)
        index = DenseIndex.load(vectors_dir)
        encoder = embeddings.Encoder(index.meta["model"], max_seq_length=index.meta["max_seq_length"])
        return cls(conn, index, encoder,
                   joblib.load(models_dir / "field_clf.joblib"),
                   lgb.Booster(model_file=str(models_dir / "ranker.txt")),
                   joblib.load(models_dir / "guard.joblib"))

    def search(self, text: str, k: int = 10) -> dict:
        start = time.perf_counter()
        built = self.builder.build([text])[0]
        t_built = time.perf_counter()
        scores = self.ranker.predict(built["X"])
        order = np.argsort(-scores, kind="stable")[:k]
        t_ranked = time.perf_counter()
        guard_x = np.append(built["query"], scores.max()).reshape(1, -1)
        prob = float(self.guard["model"].predict_proba(guard_x)[0, 1])
        end = time.perf_counter()
        dense_scores = built["X"][:, 0]
        return {
            "results": [{"id": built["ids"][i], "score": float(scores[i]), "dense_score": float(dense_scores[i])}
                        for i in order],
            "accepted": prob >= self.guard["threshold"],
            "guard_prob": round(prob, 4),
            "timings_ms": {**{key: round(v, 1) for key, v in self.builder.last_timings.items()},
                           "ranker_ms": round((t_ranked - t_built) * 1000, 1),
                           "guard_ms": round((end - t_ranked) * 1000, 1),
                           "total_ms": round((end - start) * 1000, 1)},
        }
