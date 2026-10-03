# %%
REPO_URL = "https://github.com/ismailyilmazz/arxiv-rag.git"
MODEL = "granite-311m-384"
NUM_SHARDS = 2
DB = "/kaggle/working/papers_full.db"
VEC = "/kaggle/working/vectors_full"

# %%
import glob
import os
import shutil
import subprocess
import time

!git clone {REPO_URL} /kaggle/working/arxiv-rag
%cd /kaggle/working/arxiv-rag
!pip install -q -U python-dotenv sentence-transformers

sources = glob.glob("/kaggle/input/**/papers_full.db", recursive=True)
assert sources, "papers_full.db bulunamadı. 03 notebook'unu Input olarak ekledin mi?"
for path in glob.glob(sources[0] + "*"):
    shutil.copy(path, "/kaggle/working/")
print(sources[0])

import torch
assert torch.cuda.device_count() >= NUM_SHARDS, f"{NUM_SHARDS} GPU gerekli. Settings > Accelerator: GPU T4 x2"
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %%
procs = []
for shard in range(NUM_SHARDS):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
    log = open(f"/kaggle/working/embed_shard{shard}.log", "w")
    cmd = ["python", "-m", "scripts.embed", "--model", MODEL, "--db", DB, "--out-dir", f"{VEC}_shard{shard}",
           "--shard", str(shard), "--num-shards", str(NUM_SHARDS)]
    procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))

while any(p.poll() is None for p in procs):
    time.sleep(600)
    for shard in range(NUM_SHARDS):
        lines = open(f"/kaggle/working/embed_shard{shard}.log").read().strip().splitlines()
        print(f"parça {shard}: {lines[-1] if lines else '-'}", flush=True)

for shard, p in enumerate(procs):
    assert p.returncode == 0, open(f"/kaggle/working/embed_shard{shard}.log").read()[-3000:]

# %%
SHARD_DIRS = " ".join(f"{VEC}_shard{s}" for s in range(NUM_SHARDS))
!python -m scripts.merge_vectors {SHARD_DIRS} --out-dir {VEC}
for s in range(NUM_SHARDS):
    shutil.rmtree(f"{VEC}_shard{s}")

# %%
!python -m scripts.evaluate --method dense --db {DB} --vectors-dir {VEC}
!python -m scripts.evaluate --method hybrid --db {DB} --vectors-dir {VEC}

# %%
import json

reports = [("bm25", "eval/results/bm25_3182775.json"),
           ("dense", f"eval/results/dense-{MODEL}_3182775.json"),
           ("hybrid", f"eval/results/hybrid-{MODEL}_3182775.json")]
print(f"{'yöntem':<10}{'title/en':>10}{'synth/en':>10}{'synth/tr':>10}{'mrr/tr':>9}{'ms':>9}")
for name, path in reports:
    r = json.load(open(path))
    g = r["groups"]
    print(f"{name:<10}{g['title/en']['hit@10']:>10.3f}{g['synthetic/en']['hit@10']:>10.3f}"
          f"{g['synthetic/tr']['hit@10']:>10.3f}{g['synthetic/tr']['mrr@10']:>9.3f}{r['median_latency_ms']:>9.1f}")
