# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
DB = "/kaggle/working/papers_full.db"
CACHE = "/kaggle/working/ceviri_onbellek.json"
MODES = ["replace", "fuse", "off"]
EXAMPLES = [
    "büyük dil modellerinde halüsinasyon tespiti",
    "çizge sinir ağları ile trafik tahmini",
    "kara delik birleşmesinden kütleçekim dalgaları",
    "düşük kaynaklı diller için makine çevirisi",
]

# %%
import glob
import json
import os
import shutil
import sqlite3
import statistics
import sys

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q -U python-dotenv sentence-transformers nltk lightgbm openai
sys.path.insert(0, "/kaggle/working/arxiv-rag")

from kaggle_secrets import UserSecretsClient

os.environ["LLM_API_KEY"] = UserSecretsClient().get_secret("LLM_API_KEY")


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

# %%
for mode in MODES:
    !python -m scripts.evaluation.evaluate --method pipeline --db {DB} --vectors-dir {VEC} --models-dir {MODELS} --langs tr --translation {mode} --translation-cache {CACHE}

# %%
def report(mode):
    tag = "" if mode == "off" else f"-ceviri-{mode}"
    path = [p for p in glob.glob(f"eval/results/pipeline-*{tag}-tr_*.json") if mode != "off" or "-ceviri-" not in p][0]
    return json.load(open(path, encoding="utf-8"))


reports = {mode: report(mode) for mode in ["off", "replace", "fuse"]}
ceiling = json.load(open(glob.glob("eval/results/pipeline-granite-311m-384_3182775.json")[0], encoding="utf-8"))

print(f"{'grup':<14}{'yöntem':<10}{'hit@1':>8}{'hit@10':>8}{'mrr@10':>8}")
for group in ["synthetic/tr", "manual/tr"]:
    for mode, r in reports.items():
        m = r["groups"][group]
        print(f"{group:<14}{mode:<10}{m['hit@1']:>8.3f}{m['hit@10']:>8.3f}{m['mrr@10']:>8.3f}")
m = ceiling["groups"]["synthetic/en"]
print(f"{'synthetic/en':<14}{'tavan':<10}{m['hit@1']:>8.3f}{m['hit@10']:>8.3f}{m['mrr@10']:>8.3f}  (aynı 150 hedef, İngilizce sorgu)")

print("\nBekçinin kabul oranı (offtopic düşük, diğerleri yüksek olmalı):")
for mode, r in reports.items():
    print(f"  {mode:<8}", r.get("guard_accept_rate"))

print("\nOrtanca gecikme (ms) ve çeviri maliyeti:")
for mode, r in reports.items():
    t = r.get("translation", {})
    print(f"  {mode:<8} gecikme {r['median_latency_ms']:>7}  çeviri {r.get('median_stage_ms', {}).get('translate_ms', 0):>6}  "
          f"token {t.get('tokens', 0):>6}  hata {t.get('failures', 0)}")

print("\nÖrnek çeviriler:")
for e in reports["replace"]["translation"]["examples"][:20]:
    print(f"  {e['query']}\n    -> {e['translated']}")


# %%
from arxiv_rag.search.pipeline import SearchPipeline
from arxiv_rag.search.translate import QueryTranslator
from arxiv_rag.storage.db import connect
from arxiv_rag.storage.tex import clean_tex

conn = connect(DB)
pipe = SearchPipeline.load(conn, VEC, MODELS, translation="off")
pipe.translator = QueryTranslator(cache_path=CACHE)
for query in EXAMPLES:
    print(f"\n=== {query}")
    for mode in ["off", "replace", "fuse"]:
        pipe.translation = mode
        out = pipe.search(query, k=5)
        print(f"  [{mode}] çeviri: {out['translated_query']}")
        for r in out["results"]:
            title = conn.execute("SELECT title FROM papers WHERE id = ?", (r["id"],)).fetchone()[0]
            print(f"     - {clean_tex(title)[:100]}")
