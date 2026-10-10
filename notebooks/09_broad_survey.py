# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
DB = "/kaggle/working/papers_full.db"
DRY_RUN = False
RUNS = [
    {"topic": "Mixture of experts for large language models", "type": "survey", "lang": "en",
     "seeds": ["1701.06538", "2101.03961", "2401.04088"], "gold": "2407.06204"},
    {"topic": "Retrieval-augmented generation for large language models", "type": "proposal", "lang": "tr",
     "seeds": ["2005.11401", "2004.04906"], "gold": "2312.10997"},
    {"topic": "Mixture of experts for large language models", "type": "synthesis", "lang": "en",
     "seeds": ["1701.06538", "2101.03961", "2401.04088"], "gold": "2407.06204"},
]

# %%
import glob
import os
import shutil
import sqlite3

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["LLM_API_KEY"] = secrets.get_secret("LLM_API_KEY")
os.environ["ARXIV_CONTACT_EMAIL"] = secrets.get_secret("ARXIV_CONTACT_EMAIL")
try:
    os.environ["S2_API_KEY"] = secrets.get_secret("S2_API_KEY")
    print("Semantic Scholar: anahtarla")
except Exception:
    print("Semantic Scholar: anahtarsız")
try:
    os.environ["GITHUB_TOKEN"] = secrets.get_secret("GITHUB_TOKEN")
    print("GitHub: sonuçlar repoya gönderilecek")
except Exception:
    print("GitHub: token yok, sonuçlar sadece Output'ta kalacak")

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q -U python-dotenv sentence-transformers nltk lightgbm beautifulsoup4 pypdf


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
VEC, MODELS = os.path.dirname(metas[0]), os.path.dirname(rankers[0])
conn = sqlite3.connect(DB)
for run in RUNS:
    for pid in run["seeds"] + [run["gold"]]:
        row = conn.execute("SELECT title FROM papers WHERE id = ?", (pid,)).fetchone()
        print(pid, "->", row[0] if row else "KORPUSTA YOK")

# %%
flag = "--dry-run" if DRY_RUN else ""
for run in RUNS:
    seeds, topic, gold, kind, lang = " ".join(run["seeds"]), run["topic"], run["gold"], run["type"], run["lang"]
    !python -m scripts.generate_broad --topic "{topic}" --type {kind} --lang {lang} --seeds {seeds} --gold-survey {gold} --db {DB} --vectors-dir {VEC} --models-dir {MODELS} --cache-db /kaggle/working/cache.db --out-dir /kaggle/working/generations {flag}

if not DRY_RUN:
    shutil.make_archive("/kaggle/working/generations", "zip", "/kaggle/working/generations")
    print("İndir: Output -> generations.zip", sorted(os.listdir("/kaggle/working/generations")))
    !python -m scripts.publish_results --repo-dir /kaggle/working/arxiv-rag --source-dir /kaggle/working/generations
