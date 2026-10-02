import hashlib
import json
import sys

import numpy as np
import pytest

from core import embeddings
from core.search_dense import DenseIndex
from core.search_hybrid import rrf
from scripts import embed, evaluate, make_subset

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


def test_rrf_rewards_agreement():
    fused = rrf([["a", "b"], ["b", "c"]])
    assert [pid for pid, _ in fused] == ["b", "a", "c"]


def test_normalize_gives_unit_vectors():
    v = embeddings.normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(np.linalg.norm(v[0]), 1.0) and np.all(np.isfinite(v))


def test_dense_index_returns_nearest(small_db):
    conn, _ = small_db
    pks = np.array([1, 2, 3])
    vectors = np.eye(3, DIM, dtype=np.float16)
    index = DenseIndex(vectors, pks, {"model": "e5-small"})
    query = np.zeros(DIM, dtype=np.float32)
    query[1] = 1.0
    assert index.search(conn, query, k=1)[0][0] == "2310.00002"


def test_embed_subset_and_evaluate(small_db, tmp_path, monkeypatch):
    conn, db = small_db
    monkeypatch.setattr(embeddings, "Encoder", FakeEncoder)
    queries = tmp_path / "queries.jsonl"
    rows = [
        {"qid": "a", "type": "synthetic", "lang": "en", "query": "robot grasping policies", "target_id": "2310.00002"},
        {"qid": "b", "type": "offtopic", "lang": "en", "query": "pizza near me", "target_id": None},
    ]
    queries.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    subset = tmp_path / "subset.npy"
    monkeypatch.setattr(sys, "argv", ["x", "--db", str(db), "--queries", str(queries), "--size", "2", "--out", str(subset)])
    make_subset.main()
    assert len(np.load(subset)) == 2

    vec_dir = tmp_path / "vec_subset"
    monkeypatch.setattr(sys, "argv", ["x", "--model", "e5-small", "--db", str(db), "--out-dir", str(vec_dir),
                                      "--pks", str(subset), "--chunk", "1"])
    embed.main()
    meta = json.loads((vec_dir / "meta.json").read_text())
    assert meta["count"] == 2 and meta["model"] == "e5-small"

    results = tmp_path / "results"
    monkeypatch.setattr(evaluate.config, "RESULTS_DIR", results)
    monkeypatch.setattr(sys, "argv", ["x", "--method", "dense", "--db", str(db), "--queries", str(queries),
                                      "--vectors-dir", str(vec_dir)])
    evaluate.main()
    report = json.loads((results / "dense-e5-small_2.json").read_text(encoding="utf-8"))
    assert report["groups"]["synthetic/en"]["hit@1"] == 1.0

    monkeypatch.setattr(sys, "argv", ["x", "--method", "hybrid", "--db", str(db), "--queries", str(queries),
                                      "--vectors-dir", str(vec_dir)])
    with pytest.raises(SystemExit):
        evaluate.main()

    full_dir = tmp_path / "vec_full"
    monkeypatch.setattr(sys, "argv", ["x", "--model", "e5-small", "--db", str(db), "--out-dir", str(full_dir)])
    embed.main()
    monkeypatch.setattr(sys, "argv", ["x", "--method", "hybrid", "--db", str(db), "--queries", str(queries),
                                      "--vectors-dir", str(full_dir)])
    evaluate.main()
    report = json.loads((results / "hybrid-e5-small_3.json").read_text(encoding="utf-8"))
    assert report["groups"]["synthetic/en"]["hit@1"] == 1.0
