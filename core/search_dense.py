import json
import sqlite3
from pathlib import Path

import numpy as np


class DenseIndex:
    def __init__(self, vectors: np.ndarray, pks: np.ndarray, meta: dict):
        self.vectors = vectors.astype(np.float32)
        self.pks = pks
        self.meta = meta

    @classmethod
    def load(cls, directory: Path) -> "DenseIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(np.load(directory / "vectors.npy"), np.load(directory / "pks.npy"), meta)

    def __len__(self) -> int:
        return len(self.pks)

    def _positions(self) -> np.ndarray:
        if not hasattr(self, "_pos"):
            self._pos = np.full(int(self.pks.max()) + 1, -1, dtype=np.int64)
            self._pos[self.pks] = np.arange(len(self.pks))
        return self._pos

    def scores_for(self, query_vector: np.ndarray, pks: list[int]) -> np.ndarray:
        rows = self._positions()[np.asarray(pks, dtype=np.int64)]
        return self.vectors[rows] @ query_vector.astype(np.float32)

    def search_batch(self, queries: np.ndarray, k: int, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.pks))
        out_pks = np.empty((len(queries), k), dtype=np.int64)
        out_scores = np.empty((len(queries), k), dtype=np.float32)
        for start in range(0, len(queries), chunk):
            scores = queries[start:start + chunk].astype(np.float32) @ self.vectors.T
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1)
            top = np.take_along_axis(top, order, axis=1)
            out_pks[start:start + len(top)] = self.pks[top]
            out_scores[start:start + len(top)] = np.take_along_axis(top_scores, order, axis=1)
        return out_pks, out_scores

    def search(self, conn: sqlite3.Connection, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        scores = self.vectors @ query_vector.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        pks = [int(p) for p in self.pks[top]]
        rows = conn.execute(f"SELECT pk, id FROM papers WHERE pk IN ({','.join('?' * len(pks))})", pks)
        id_of = {r[0]: r[1] for r in rows}
        return [(id_of[pk], float(scores[i])) for pk, i in zip(pks, top) if pk in id_of]
