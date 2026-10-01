import argparse
import json
import random
import re
import statistics
from pathlib import Path

from core import config, llm
from core.db import connect
from core.splits import split_of

SYNTH_PROMPT = """Paper title: {title}
Abstract: {abstract}

Imagine a researcher who has NOT read this paper but is looking for work like it.
Write the short search query (3 to 10 words) they would type into an academic search engine.
Rules:
- Describe the problem and the approach in general terms. Do not copy distinctive phrases from the title.
- No quotes, no author names.
Then write the same query in Turkish, the way a Turkish researcher would write it:
- Keep technical terms that Turkish researchers normally use in English as they are (for example SLAM, transformer, Banach, Bayesian).
- Otherwise use the standard Turkish scientific term. Do not invent words.
Return JSON: {{"en": "<English query>", "tr": "<Turkish query>"}}"""

OFFTOPIC_PROMPT = """Write {n} different everyday questions that people type into a search box and that have nothing to do with scientific research.
Cover many topics: cooking, sports, travel, shopping, personal finance, entertainment, local services, health tips, home repair.
Write them in {language}. Return JSON: {{"questions": ["...", "..."]}}"""


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


def append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def title_overlap(query: str, title: str) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    t = set(re.findall(r"\w+", title.lower()))
    return len(q & t) / len(q) if q else 0.0


def valid_synthetic(item: dict, title: str) -> bool:
    en, tr = str(item.get("en", "")).strip(), str(item.get("tr", "")).strip()
    return (
        bool(en) and bool(tr)
        and len(en.split()) <= 20 and len(tr.split()) <= 20
        and en.lower() != title.lower()
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--out", type=Path, default=config.EVAL_QUERIES_PATH)
    p.add_argument("--n-title", type=int, default=300)
    p.add_argument("--n-synthetic", type=int, default=150)
    p.add_argument("--n-offtopic", type=int, default=50, help="dil başına")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic-model", default=config.LLM_MODEL,
                   help="Türkçe teknik terimler için ana model daha güvenilir")
    p.add_argument("--redo-synthetic", action="store_true",
                   help="sadece sentetik satırları silip yeniden üretir")
    args = p.parse_args()

    if args.n_synthetic or args.n_offtopic:
        llm._client()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.redo_synthetic and args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            kept = [line for line in f if line.strip() and json.loads(line)["type"] != "synthetic"]
        args.out.write_text("".join(kept), encoding="utf-8")
        print(f"Sentetik satırlar silindi, {len(kept)} satır kaldı.")
    done = load_done(args.out)
    conn = connect(args.db)

    pool = sorted(r["id"] for r in conn.execute("SELECT id FROM papers") if split_of(r["id"]) == "test")
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    print(f"Test havuzunda {len(pool):,} makale var.")

    def paper(pid):
        return conn.execute("SELECT title, abstract FROM papers WHERE id = ?", (pid,)).fetchone()

    for pid in pool[: args.n_title]:
        qid = f"title-{pid}"
        if qid not in done:
            append(args.out, {"qid": qid, "type": "title", "lang": "en",
                              "query": paper(pid)["title"], "target_id": pid})

    synth_ids = pool[: args.n_synthetic]
    for i, pid in enumerate(synth_ids, start=1):
        if f"synthetic-en-{pid}" in done:
            continue
        row = paper(pid)
        try:
            item = llm.chat_json(
                SYNTH_PROMPT.format(title=row["title"], abstract=row["abstract"][:1500]),
                model=args.synthetic_model,
            )
        except Exception as e:
            print(f"  atlandı {pid}: {e}")
            continue
        if not valid_synthetic(item, row["title"]):
            print(f"  geçersiz cevap, atlandı: {pid}")
            continue
        for lang in ("en", "tr"):
            append(args.out, {"qid": f"synthetic-{lang}-{pid}", "type": "synthetic", "lang": lang,
                              "query": item[lang].strip(), "target_id": pid})
        print(f"  sentetik {i}/{len(synth_ids)}", end="\r")

    for lang, language in (("en", "English"), ("tr", "Turkish")):
        if args.n_offtopic == 0 or f"offtopic-{lang}-0" in done:
            continue
        item = llm.chat_json(OFFTOPIC_PROMPT.format(n=args.n_offtopic, language=language),
                             model=config.LLM_MODEL_FAST)
        questions = list(dict.fromkeys(q.strip() for q in item.get("questions", []) if q.strip()))
        for j, q in enumerate(questions[: args.n_offtopic]):
            append(args.out, {"qid": f"offtopic-{lang}-{j}", "type": "offtopic", "lang": lang,
                              "query": q, "target_id": None})

    with open(args.out, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    counts = {}
    for r in rows:
        key = f"{r['type']}/{r['lang']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"\n{args.out} hazır: {len(rows)} sorgu  {counts}")
    titles = {r["target_id"]: r["query"] for r in rows if r["type"] == "title"}
    overlaps = [title_overlap(r["query"], titles[r["target_id"]]) for r in rows
                if r["type"] == "synthetic" and r["lang"] == "en" and r["target_id"] in titles]
    if overlaps:
        print(f"İngilizce sentetik sorgularda ortalama başlık örtüşmesi: {statistics.mean(overlaps):.2f}")


if __name__ == "__main__":
    main()
