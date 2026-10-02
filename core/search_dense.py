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

    def search(self, conn: sqlite3.Connection, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        scores = self.vectors @ query_vector.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        pks = [int(p) for p in self.pks[top]]
        rows = conn.execute(f"SELECT pk, id FROM papers WHERE pk IN ({','.join('?' * len(pks))})", pks)
        id_of = {r[0]: r[1] for r in rows}
        return [(id_of[pk], float(scores[i])) for pk, i in zip(pks, top) if pk in id_of]
