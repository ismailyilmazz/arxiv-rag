# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
DB = "/kaggle/working/papers_full.db"
MODELS = "/kaggle/working/models"
FEATS = "/kaggle/working/features"

# %%
import glob
import os
import shutil
import sqlite3

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q -U python-dotenv sentence-transformers nltk lightgbm


def has_term_df(path):
    try:
        with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as c:
            return c.execute("SELECT 1 FROM sqlite_master WHERE name = 'term_df'").fetchone() is not None
    except sqlite3.Error:
        return False


dbs = [p for p in glob.glob("/kaggle/input/**/papers_full.db", recursive=True) if has_term_df(p)]
metas = glob.glob("/kaggle/input/**/vectors_full/meta.json", recursive=True)
assert dbs, "term_df tablolu papers_full.db bulunamadı. 05 notebook'unu Input olarak ekledin mi?"
assert metas, "vectors_full bulunamadı. 04 notebook'unu Input olarak ekledin mi?"
assert os.path.exists("train/queries.jsonl") and os.path.exists("train/offtopic.jsonl"), \
    "train/ klasörü repoda yok. Adım 4a çıktısını push'ladın mı?"
for path in glob.glob(dbs[0] + "*"):
    shutil.copy(path, "/kaggle/working/")
VEC = os.path.dirname(metas[0])
print(dbs[0])
print(VEC)

# %%
!python -m scripts.train_field_classifier --db {DB} --vectors-dir {VEC} --out {MODELS}/field_clf.joblib

# %%
COMMON = f"--db {DB} --vectors-dir {VEC} --field-model {MODELS}/field_clf.joblib --out-dir {FEATS}"
!python -m scripts.build_features {COMMON} --queries train/queries.jsonl --split train --add-titles
!python -m scripts.build_features {COMMON} --queries train/offtopic.jsonl --split offtopic
!python -m scripts.build_features {COMMON} --queries eval/queries.jsonl --split eval

# %%
!python -m scripts.train_ranker --features-dir {FEATS} --model-out {MODELS}/ranker.txt
!python -m scripts.train_guard --features-dir {FEATS} --ranker-model {MODELS}/ranker.txt --model-out {MODELS}/guard.joblib

# %%
!python -m scripts.evaluate --method bm25-gate --db {DB}
