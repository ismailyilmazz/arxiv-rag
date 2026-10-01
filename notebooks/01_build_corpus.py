# %%
import glob
import json
import random
import re
from collections import Counter
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

DEV_SAMPLE_SIZE = 30_000
CHUNK = 200_000
SEED = 42
OUT = "/kaggle/working"

paths = glob.glob("/kaggle/input/**/arxiv-metadata-oai-snapshot.json", recursive=True)
assert paths, "Snapshot bulunamadı. Sağ panelden Cornell-University/arxiv veri setini ekle."
SNAPSHOT = paths[0]
print("Snapshot:", SNAPSHOT)

SCHEMA = pa.schema([(name, pa.string()) for name in (
    "id", "title", "abstract", "authors", "categories",
    "primary_category", "published", "updated", "license", "doi",
)])

# %%
_WS = re.compile(r"\s+")

def clean(text):
    return _WS.sub(" ", text or "").strip()

def first_version_date(versions):
    try:
        created = versions[0]["created"]
        return datetime.strptime(created, "%a, %d %b %Y %H:%M:%S %Z").date().isoformat()
    except (IndexError, KeyError, TypeError, ValueError):
        return None

# %%
rng = random.Random(SEED)
writer = pq.ParquetWriter(f"{OUT}/corpus.parquet", SCHEMA)
buffer, dev_sample = [], []
seen_ids = set()
kept = total = skipped_dup = skipped_empty = 0
by_primary, by_year, by_group = Counter(), Counter(), Counter()

def flush():
    if buffer:
        writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
        buffer.clear()

with open(SNAPSHOT, encoding="utf-8") as f:
    for line in f:
        total += 1
        r = json.loads(line)
        pid = r["id"]
        if pid in seen_ids:
            skipped_dup += 1
            continue
        title, abstract = clean(r.get("title")), clean(r.get("abstract"))
        cats = (r.get("categories") or "").split()
        if not title or not abstract or not cats:
            skipped_empty += 1
            continue
        seen_ids.add(pid)

        published = first_version_date(r.get("versions"))
        row = {
            "id": pid,
            "title": title,
            "abstract": abstract,
            "authors": clean(r.get("authors")),
            "categories": " ".join(cats),
            "primary_category": cats[0],
            "published": published,
            "updated": r.get("update_date"),
            "license": r.get("license"),
            "doi": r.get("doi"),
        }
        buffer.append(row)
        kept += 1
        by_primary[cats[0]] += 1
        by_group[cats[0].split(".")[0]] += 1
        by_year[published[:4] if published else "bilinmiyor"] += 1

        if len(dev_sample) < DEV_SAMPLE_SIZE:
            dev_sample.append(row)
        else:
            j = rng.randrange(kept)
            if j < DEV_SAMPLE_SIZE:
                dev_sample[j] = row

        if len(buffer) >= CHUNK:
            flush()
            print(f"  {kept:,} makale yazıldı", end="\r")

flush()
writer.close()
pq.write_table(pa.Table.from_pylist(dev_sample, schema=SCHEMA), f"{OUT}/dev_sample.parquet")

print(f"\nSnapshot'taki kayıt: {total:,}")
print(f"Korpusa giren     : {kept:,}  (tekrar: {skipped_dup:,}, boş: {skipped_empty:,})")

# %%
stats = {
    "snapshot_records": total,
    "corpus_records": kept,
    "skipped_duplicate": skipped_dup,
    "skipped_empty": skipped_empty,
    "by_group": dict(by_group.most_common()),
    "top_primary_categories": dict(by_primary.most_common(40)),
    "by_year": dict(sorted(by_year.items())),
    "estimated_embedding_gb_384d_float16": round(kept * 384 * 2 / 1e9, 2),
}
with open(f"{OUT}/stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(json.dumps({k: stats[k] for k in ("corpus_records", "by_group",
                                         "estimated_embedding_gb_384d_float16")},
                 ensure_ascii=False, indent=2))
