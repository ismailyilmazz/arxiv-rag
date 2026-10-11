import hashlib

import numpy as np

from arxiv_rag.search import embeddings

DIM = 64


class FakeEncoder:
    def __init__(self, name, device=None, max_seq_length=512):
        self.name = name
        self.spec = embeddings.MODELS[name]

    def _encode(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().replace(".", " ").split():
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1
        return embeddings.normalize(out + 1e-6)

    def encode_queries(self, texts, batch_size=64):
        return self._encode(texts)

    def encode_passages(self, texts, batch_size=128):
        return self._encode(texts)
