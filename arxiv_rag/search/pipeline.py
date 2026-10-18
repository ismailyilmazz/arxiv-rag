import sqlite3
import time
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np

from arxiv_rag import config
from arxiv_rag.search import embeddings
from arxiv_rag.search.dense import DenseIndex
from arxiv_rag.search.features import FeatureBuilder

TRANSLATION_MODES = ("off", "replace", "fuse")


def fuse(rankings: list[list[dict]], k: int, c: int = 60) -> list[dict]:
    scores, first = {}, {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item["id"]] = scores.get(item["id"], 0.0) + 1.0 / (c + rank)
            first.setdefault(item["id"], item)
    order = sorted(scores, key=lambda pid: -scores[pid])[:k]
    return [{**first[pid], "score": round(scores[pid], 6)} for pid in order]


class SearchPipeline:
    def __init__(
        self,
        conn: sqlite3.Connection,
        index: DenseIndex,
        encoder,
        field_model: dict,
        ranker: lgb.Booster,
        guard: dict,
        depth: int = 100,
        gate: int = 2,
        translator=None,
        translation: str = "off",
    ):
        if translation not in TRANSLATION_MODES:
            raise ValueError(f"Bilinmeyen çeviri modu: {translation}")
        self.index = index
        self.ranker = ranker
        self.guard = guard
        self.translator = translator
        self.translation = translation if translator is not None else "off"
        self._parts = (index, encoder, field_model, ranker, guard, depth, gate)
        self.builder = FeatureBuilder(conn, index, encoder, field_model, depth=depth, gate=gate)

    def bind(self, conn: sqlite3.Connection) -> "SearchPipeline":
        return SearchPipeline(conn, *self._parts, translator=self.translator, translation=self.translation)

    @classmethod
    def load(
        cls, conn: sqlite3.Connection, vectors_dir: Path, models_dir: Path, translation: Optional[str] = None
    ) -> "SearchPipeline":
        models_dir = Path(models_dir)
        translation = translation or config.QUERY_TRANSLATION
        translator = None
        if translation != "off" and config.LLM_API_KEY:
            from arxiv_rag.search.translate import QueryTranslator

            translator = QueryTranslator()
        index = DenseIndex.load(vectors_dir)
        encoder = embeddings.Encoder(index.meta["model"], max_seq_length=index.meta["max_seq_length"])
        return cls(
            conn,
            index,
            encoder,
            joblib.load(models_dir / "field_clf.joblib"),
            lgb.Booster(model_file=str(models_dir / "ranker.txt")),
            joblib.load(models_dir / "guard.joblib"),
            translator=translator,
            translation=translation,
        )

    def search(self, text: str, k: int = 10) -> dict:
        start = time.perf_counter()
        translated = self.translator(text) if self.translation != "off" else None
        translate_ms = round((time.perf_counter() - start) * 1000, 1)
        if not translated:
            out = self._search(text, k)
        elif self.translation == "replace":
            out = self._search(translated, k)
        else:
            original, english = self._search(text, k * 3), self._search(translated, k * 3)
            out = {**english, "results": fuse([original["results"], english["results"]], k)}
            out["timings_ms"] = {
                key: round(original["timings_ms"].get(key, 0) + english["timings_ms"].get(key, 0), 1)
                for key in english["timings_ms"]
            }
        out["translated_query"] = translated
        out["timings_ms"]["translate_ms"] = translate_ms
        return out

    def _search(self, text: str, k: int) -> dict:
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
            "results": [
                {"id": built["ids"][i], "score": float(scores[i]), "dense_score": float(dense_scores[i])} for i in order
            ],
            "accepted": prob >= self.guard["threshold"],
            "guard_prob": round(prob, 4),
            "timings_ms": {
                **{key: round(v, 1) for key, v in self.builder.last_timings.items()},
                "ranker_ms": round((t_ranked - t_built) * 1000, 1),
                "guard_ms": round((end - t_ranked) * 1000, 1),
                "total_ms": round((end - start) * 1000, 1),
            },
        }
