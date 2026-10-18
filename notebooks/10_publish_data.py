REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
HF_REPO = "ismailylmz283/arxiv-rag-data"
PRIVATE = False
SERVING_SKLEARN = "1.6.1"

# %%
!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag

# %%
import datetime
import glob
import json
import os
import sqlite3
import warnings

import joblib
import lightgbm
import sklearn
from huggingface_hub import CommitOperationAdd, HfApi
from kaggle_secrets import UserSecretsClient
from sklearn.exceptions import InconsistentVersionWarning


def has_term_df(path):
    try:
        with sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True) as c:
            return c.execute("SELECT 1 FROM sqlite_master WHERE name = 'term_df'").fetchone() is not None
    except sqlite3.Error as error:
        print(f"Açılamadı: {path} ({error})")
        return False


def saved_sklearn_version(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        joblib.load(path)
    for w in caught:
        if issubclass(w.category, InconsistentVersionWarning):
            return w.message.original_sklearn_version
    return sklearn.__version__


print("Kaggle girdileri:")
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root[len("/kaggle/input"):].count(os.sep)
    print("  " * depth + (os.path.basename(root) or "/kaggle/input") + "/" + (f"  ({len(files)} dosya)" if files else ""))
    if depth >= 3:
        dirs[:] = []

all_dbs = glob.glob("/kaggle/input/**/papers_full.db", recursive=True)
dbs = [p for p in all_dbs if has_term_df(p)]
metas = glob.glob("/kaggle/input/**/vectors_full/meta.json", recursive=True)
rankers = glob.glob("/kaggle/input/**/models/ranker.txt", recursive=True)
print("papers_full.db bulunanlar:", all_dbs, "| term_df içerenler:", dbs)
missing = [name for name, found in (("term_df'li papers_full.db (06'nın çıktısı)", dbs),
                                    ("vectors_full/meta.json (04'ün çıktısı)", metas),
                                    ("models/ranker.txt (06'nın çıktısı)", rankers)) if not found]
assert not missing, "Bulunamadı: " + "; ".join(missing) + ". Yukarıdaki girdi listesini kontrol et."
DB, VEC, MODELS = dbs[0], os.path.dirname(metas[0]), os.path.dirname(rankers[0])

UPLOADS = {"papers.db": DB}
for name in ("vectors.npy", "pks.npy", "meta.json"):
    UPLOADS[f"vectors/{name}"] = os.path.join(VEC, name)
for name in sorted(os.listdir(MODELS)):
    if os.path.isfile(os.path.join(MODELS, name)):
        UPLOADS[f"models/{name}"] = os.path.join(MODELS, name)
for rel, path in UPLOADS.items():
    print(f"{rel:28} {os.path.getsize(path) / 1e9:8.3f} GB  <- {path}")

versions = {name: saved_sklearn_version(path) for name, path in UPLOADS.items() if name.endswith(".joblib")}
print("Modellerin kaydedildiği scikit-learn sürümleri:", versions)
assert set(versions.values()) == {SERVING_SKLEARN}, (
    f"Modeller {versions} ile kaydedilmiş, sunucu {SERVING_SKLEARN} kullanıyor. pyproject.toml'daki sürümü eşitle."
)

# %%
with sqlite3.connect(f"file:{DB}?mode=ro&immutable=1", uri=True) as c:
    papers = c.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
meta = json.load(open(os.path.join(VEC, "meta.json")))
manifest = {
    "files": {rel: os.path.getsize(path) for rel, path in UPLOADS.items()},
    "papers": papers,
    "embedding_model": meta.get("model"),
    "scikit_learn": SERVING_SKLEARN,
    "lightgbm": lightgbm.__version__,
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
}
json.dump(manifest, open("/kaggle/working/manifest.json", "w"), indent=2)
open("/kaggle/working/README.md", "w").write(f"""---
license: cc0-1.0
pretty_name: arxiv-rag serving data
---

# arxiv-rag serving data

Serving data for [arxiv-rag](https://github.com/ismailyilmazz/arxiv-rag), an arXiv research assistant.

- `papers.db`: SQLite with FTS5 and the `term_df` table; {papers:,} arXiv papers (metadata from the Cornell arXiv dataset, CC0).
- `vectors/`: {meta.get("model")} embeddings (float16) and the row keys they belong to.
- `models/`: field classifier and guard (scikit-learn {SERVING_SKLEARN}), LightGBM ranker.
- `manifest.json`: file sizes and versions, checked by `scripts/deploy/fetch_data.py` after download.
""")
print(json.dumps(manifest, indent=2))

# %%
api = HfApi(token=UserSecretsClient().get_secret("HF_TOKEN"))
api.create_repo(HF_REPO, repo_type="dataset", private=PRIVATE, exist_ok=True)
operations = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=path) for rel, path in UPLOADS.items()]
operations += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=f"/kaggle/working/{name}") for name in ("manifest.json", "README.md")]
commit = api.create_commit(HF_REPO, operations=operations, repo_type="dataset", commit_message="Sunucu verisi")
print("Commit:", commit.commit_url)

# %%
remote = {f.rfilename: f.size for f in api.dataset_info(HF_REPO, files_metadata=True).siblings}
problems = [rel for rel, size in manifest["files"].items() if remote.get(rel) != size]
print("Uzak depodaki boyutlar manifestle aynı." if not problems else f"Farklı ya da eksik: {problems}")