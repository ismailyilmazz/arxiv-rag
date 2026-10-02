# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
MODELS = ["e5-small", "granite-97m", "granite-311m-384"]
SUBSET_SIZE = 200_000
DB = "/kaggle/working/papers_full.db"

# %%
import glob

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q -U python-dotenv sentence-transformers

corpus = glob.glob("/kaggle/input/**/corpus.parquet", recursive=True)
assert corpus, "corpus.parquet bulunamadı. Adım 1 notebook'unu Input olarak ekledin mi?"
CORPUS = corpus[0]
print(CORPUS)

import torch
assert torch.cuda.is_available(), "GPU kapalı. Settings > Accelerator: GPU T4 x2"
print(torch.cuda.get_device_name(0))

# %%
!python -m scripts.load_sqlite {CORPUS} --db {DB}

# %%
!python -m scripts.make_subset --db {DB} --size {SUBSET_SIZE} --out /kaggle/working/subset_pks.npy

# %%
for m in MODELS:
    !python -m scripts.embed --model {m} --db {DB} --pks /kaggle/working/subset_pks.npy --out-dir /kaggle/working/vectors_subset_{m}
    !python -m scripts.evaluate --method dense --db {DB} --vectors-dir /kaggle/working/vectors_subset_{m}

# %%
import json

print(f"{'model':<20}{'title/en':>10}{'synth/en':>10}{'synth/tr':>10}{'ms':>8}")
for path in sorted(glob.glob("eval/results/dense-*.json")):
    r = json.load(open(path))
    g = r["groups"]
    print(f"{r['model']:<20}{g['title/en']['hit@10']:>10.3f}{g['synthetic/en']['hit@10']:>10.3f}"
          f"{g['synthetic/tr']['hit@10']:>10.3f}{r['median_latency_ms']:>8.1f}")
