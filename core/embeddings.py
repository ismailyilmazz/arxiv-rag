from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    query_prefix: str = ""
    passage_prefix: str = ""
    truncate_dim: Optional[int] = None


MODELS = {
    "e5-small": ModelSpec("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "granite-97m": ModelSpec("ibm-granite/granite-embedding-97m-multilingual-r2"),
    "granite-311m-384": ModelSpec("ibm-granite/granite-embedding-311m-multilingual-r2", truncate_dim=384),
}


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def passage_text(title: str, abstract: str) -> str:
    return f"{title}. {abstract}"


class Encoder:
    def __init__(self, name: str, device: Optional[str] = None, max_seq_length: int = 512):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.spec = MODELS[name]
        self.model = SentenceTransformer(self.spec.hf_id, device=device, truncate_dim=self.spec.truncate_dim)
        self.model.max_seq_length = max_seq_length
        if self.model.device.type == "cuda":
            self.model.half()

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        return normalize(vectors)

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts], batch_size)

    def encode_passages(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self._encode([self.spec.passage_prefix + t for t in texts], batch_size)
