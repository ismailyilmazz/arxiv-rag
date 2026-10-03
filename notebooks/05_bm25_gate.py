# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
DB = "/kaggle/working/papers_full.db"
METHODS = ["bm25", "bm25-gate", "dense", "hybrid", "hybrid-gate", "lang-rule"]

# %%
import glob
import json
import os
import shutil
import sqlite3

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q -U python-dotenv sentence-transformers nltk

dbs = glob.glob("/kaggle/input/**/papers_full.db", recursive=True)
metas = glob.glob("/kaggle/input/**/vectors_full/meta.json", recursive=True)
assert dbs and metas, "papers_full.db veya vectors_full bulunamadı. 04 notebook'unu Input olarak ekledin mi?"
for path in glob.glob(dbs[0] + "*"):
    shutil.copy(path, "/kaggle/working/")
VEC = os.path.dirname(metas[0])
MODEL = json.load(open(metas[0]))["model"]
print(dbs[0])
print(VEC, MODEL)

# %%
!python -m scripts.build_term_df --db {DB}

# %%
for m in METHODS:
    extra = "" if m.startswith("bm25") else f"--vectors-dir {VEC}"
    !python -m scripts.evaluate --method {m} --db {DB} {extra}

# %%
N = sqlite3.connect(DB).execute("SELECT COUNT(*) FROM papers").fetchone()[0]
print(f"{'yöntem':<13}{'title/en':>10}{'synth/en':>10}{'synth/tr':>10}{'manual/tr':>11}{'ms':>9}")
for m in METHODS:
    label = m if m.startswith("bm25") else f"{m}-{MODEL}"
    r = json.load(open(f"eval/results/{label}_{N}.json"))
    g = r["groups"]
    print(f"{m:<13}{g['title/en']['hit@10']:>10.3f}{g['synthetic/en']['hit@10']:>10.3f}"
          f"{g['synthetic/tr']['hit@10']:>10.3f}{g['manual/tr']['hit@10']:>11.3f}{r['median_latency_ms']:>9.1f}")
