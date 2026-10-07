# %%
OUT = "/kaggle/working/local_bundle"

# %%
import glob
import json
import os
import shutil
import sqlite3

import numpy as np
import pyarrow.parquet as pq

dev = glob.glob("/kaggle/input/**/dev_sample.parquet", recursive=True)
dbs = glob.glob("/kaggle/input/**/papers_full.db", recursive=True)
metas = glob.glob("/kaggle/input/**/vectors_full/meta.json", recursive=True)
rankers = glob.glob("/kaggle/input/**/models/ranker.txt", recursive=True)
assert dev, "dev_sample.parquet bulunamadı. 01 notebook'unu Input olarak ekledin mi?"
assert metas, "vectors_full bulunamadı. 04 notebook'unu Input olarak ekledin mi?"
assert dbs and rankers, "Veritabanı ya da modeller bulunamadı. 06 notebook'unu Input olarak ekledin mi?"

ids = pq.read_table(dev[0], columns=["id"]).column("id").to_pylist()
conn = sqlite3.connect(f"file:{dbs[0]}?mode=ro&immutable=1", uri=True)
pk_of = {}
for i in range(0, len(ids), 20_000):
    chunk = ids[i:i + 20_000]
    pk_of.update(conn.execute(f"SELECT id, pk FROM papers WHERE id IN ({','.join('?' * len(chunk))})", chunk))

VEC = os.path.dirname(metas[0])
pks = np.load(f"{VEC}/pks.npy")
vectors = np.load(f"{VEC}/vectors.npy", mmap_mode="r")
pos = np.full(int(pks.max()) + 1, -1, dtype=np.int64)
pos[pks] = np.arange(len(pks))
pairs = sorted((int(pos[pk_of[pid]]), pid) for pid in ids if pid in pk_of and pos[pk_of[pid]] >= 0)
rows = np.array([r for r, _ in pairs])
kept = np.array([pid for _, pid in pairs])
print(f"{len(kept):,} / {len(ids):,} makalenin vektörü bulundu")

# %%
os.makedirs(OUT, exist_ok=True)
np.save(f"{OUT}/vectors.npy", np.asarray(vectors[rows], dtype=np.float16))
np.save(f"{OUT}/ids.npy", kept)
meta = json.load(open(metas[0]))
meta["count"] = int(len(kept))
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
shutil.copytree(os.path.dirname(rankers[0]), f"{OUT}/models", dirs_exist_ok=True)
shutil.make_archive("/kaggle/working/local_bundle", "zip", OUT)
print(f"local_bundle.zip: {os.path.getsize('/kaggle/working/local_bundle.zip') / 1e6:.1f} MB")
print(sorted(os.listdir(OUT)), sorted(os.listdir(f"{OUT}/models")))
