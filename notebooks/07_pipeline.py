# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
DB = "/kaggle/working/papers_full.db"
EXAMPLES = [
    "lora ile büyük dil modeli finetune etme",
    "graph neural networks for molecule property prediction",
    "kara delik birleşmesi kütleçekim dalgası",
    "en iyi pizza tarifi nasıl yapılır",
]

# %%
import glob
import os
import shutil
import sqlite3
import sys

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q -U python-dotenv sentence-transformers nltk lightgbm
sys.path.insert(0, "/kaggle/working/arxiv-rag")


def has_term_df(path):
    try:
        with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as c:
            return c.execute("SELECT 1 FROM sqlite_master WHERE name = 'term_df'").fetchone() is not None
    except sqlite3.Error:
        return False


dbs = [p for p in glob.glob("/kaggle/input/**/papers_full.db", recursive=True) if has_term_df(p)]
metas = glob.glob("/kaggle/input/**/vectors_full/meta.json", recursive=True)
rankers = glob.glob("/kaggle/input/**/models/ranker.txt", recursive=True)
assert dbs and rankers, "Veritabanı ya da modeller bulunamadı. 06 notebook'unu Input olarak ekledin mi?"
assert metas, "vectors_full bulunamadı. 04 notebook'unu Input olarak ekledin mi?"
for path in glob.glob(dbs[0] + "*"):
    shutil.copy(path, "/kaggle/working/")
VEC = os.path.dirname(metas[0])
MODELS = os.path.dirname(rankers[0])
print(dbs[0])
print(VEC)
print(MODELS)

# %%
!python -m scripts.evaluate --method pipeline --db {DB} --vectors-dir {VEC} --models-dir {MODELS}

# %%
from core.db import connect
from core.pipeline import SearchPipeline

conn = connect(DB)
pipe = SearchPipeline.load(conn, VEC, MODELS)
for text in EXAMPLES:
    out = pipe.search(text, k=3)
    karar = "kaynak yeterli" if out["accepted"] else "yeterli kaynak bulunamadı"
    print(f"\n{text}\n  bekçi: {karar} (olasılık {out['guard_prob']}), toplam {out['timings_ms']['total_ms']} ms")
    for r in out["results"]:
        title = conn.execute("SELECT title FROM papers WHERE id = ?", (r["id"],)).fetchone()[0]
        print(f"  https://arxiv.org/abs/{r['id']}  {title[:90]}")
