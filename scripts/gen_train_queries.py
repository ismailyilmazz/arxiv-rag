import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from core import config, llm
from core.db import connect
from core.search_bm25 import _stem, query_words
from core.splits import split_of

PAPER_PROMPT = """Paper title: {title}
Abstract: {abstract}

Write three search queries that would lead someone to this paper. Each query is 3 to 10 words, has no quotes, no author names, and does not copy the title.
1. "en": how an English-speaking researcher who has NOT read the paper would describe what they are looking for.
2. "tr": the same need in natural Turkish. Keep technical terms that Turkish researchers normally use in English as they are; otherwise use the standard Turkish scientific term. Do not invent words.
3. "mixed": how a Turkish software engineer would actually type it into a search box: short, lowercase, mixing Turkish words with English technical terms and method names. It MUST contain at least two Turkish words; never write it fully in English. Examples of this style, for other papers:
- gradient boosting kategorik değişkenler catboost
- knowledge distillation ile küçük model eğitme
- graph neural network node classification gcn
- vektör veritabanı ivf ile hızlı arama
- spin glass monte carlo simülasyon faz geçişi
Return JSON: {{"en": "...", "tr": "...", "mixed": "..."}}"""

OFFTOPIC_PROMPT = """Write {n} different questions that people type into a search box about {topic}.
None of them may be about scientific research. Vary the length and phrasing.
Write them in {language}. Return JSON: {{"questions": ["...", "..."]}}"""

TOPICS = [
    "cooking and recipes", "sports and fitness", "travel and holidays", "shopping and product prices",
    "personal finance and banking", "movies, series and music", "health tips and daily wellbeing",
    "home repair and cleaning", "government paperwork and local services", "relationships and daily life",
]


class ModelPool:
    MAX_STRIKES = 3

    def __init__(self, models: list[str], min_interval: float):
        self.models = list(models)
        self.min_interval = min_interval
        self.next_free = {m: 0.0 for m in models}
        self.tokens = Counter()
        self.exhausted: set[str] = set()
        self.daily_exhausted: set[str] = set()
        self.strikes = Counter()
        self.bad = Counter()
        self.caps: dict[str, int] = {}

    def _limit_hit(self, model: str, error: Exception) -> None:
        self.strikes[model] += 1
        if self.strikes[model] == 1:
            print(f"\n  {model}: hız sınırı hatası: {str(error)[:300]}")
        if self.strikes[model] >= self.MAX_STRIKES:
            print(f"\n  {model}: üst üste {self.MAX_STRIKES} kez hız sınırına takıldı, bu çalıştırmada devre dışı.")
            self.exhausted.add(model)
        else:
            self.next_free[model] = time.monotonic() + 30

    def _bad_answer(self, model: str, error: Exception) -> None:
        self.bad[model] += 1
        if self.bad[model] == 1:
            print(f"\n  {model}: kullanılamayan cevap: {type(error).__name__}: {str(error)[:200]}")
        if self.bad[model] >= self.MAX_STRIKES:
            print(f"\n  {model}: üst üste {self.MAX_STRIKES} kullanılamayan cevap, bu çalıştırmada devre dışı.")
            self.exhausted.add(model)

    def inherit_account_limits(self, other: "ModelPool") -> None:
        self.daily_exhausted |= other.daily_exhausted
        self.exhausted |= other.daily_exhausted
        self.caps.update(other.caps)

    def call(self, prompt: str, max_tokens: int = 1024) -> tuple[dict, str]:
        attempts = 0
        while True:
            active = [m for m in self.models if m not in self.exhausted]
            if not active:
                raise RuntimeError("all-exhausted")
            attempts += 1
            if attempts > 4 * len(self.models):
                raise RuntimeError("too-many-attempts")
            model = min(active, key=lambda m: self.next_free[m])
            wait = self.next_free[model] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.next_free[model] = time.monotonic() + self.min_interval
            tokens = min(max_tokens, self.caps.get(model, max_tokens))
            try:
                data, used = llm.chat(prompt, model=model, max_retries=1, max_tokens=tokens)
            except Exception as e:
                cap = llm.output_limit(e)
                if cap is not None and tokens > cap // 4:
                    self.caps[model] = max(64, cap // 4)
                    self.next_free[model] = 0.0
                    print(f"\n  {model}: çıktı sınırı {cap}, istek {self.caps[model]} token'a düşürüldü.")
                    continue
                kind = llm.limit_kind(e)
                if kind == "day":
                    print(f"\n  {model}: günlük kota doldu, diğer modellerle devam.")
                    self.exhausted.add(model)
                    self.daily_exhausted.add(model)
                    continue
                if kind == "minute":
                    self._limit_hit(model, e)
                    continue
                self._bad_answer(model, e)
                continue
            self.strikes[model] = 0
            self.bad[model] = 0
            self.tokens[model] += used
            return data, model


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def new_words(query: str, en: str) -> list[str]:
    en_stems = {_stem(w) for w in query_words(en)}
    return [w for w in query_words(query) if _stem(w) not in en_stems]


def clean_query(value, title: str) -> str:
    text = " ".join(str(value or "").split())
    if not text or len(text.split()) > 20 or text.lower() == title.lower():
        return ""
    return text


def generate_offtopic(pool: ModelPool, out: Path, total: int, eval_questions: set[str]) -> None:
    if total <= 0:
        return
    done = read_jsonl(out)
    seen = {r["query"].lower() for r in done} | eval_questions
    per_call = 25
    parts = max(1, -(-total // (per_call * len(TOPICS) * 2)))
    jobs = [(topic, lang, part) for part in range(parts) for topic in TOPICS for lang in ("en", "tr")]
    per_job = max(1, min(per_call, -(-total // len(jobs))))
    finished = {r["job"] for r in done}
    for i, (topic, lang, part) in enumerate(jobs):
        job = f"{topic}|{lang}|{part}"
        if job in finished:
            continue
        language = "English" if lang == "en" else "Turkish"
        try:
            data, model = pool.call(OFFTOPIC_PROMPT.format(n=per_job, topic=topic, language=language), max_tokens=1500)
        except RuntimeError as e:
            if str(e) == "too-many-attempts":
                continue
            return
        rows = []
        for q in data.get("questions", []):
            q = " ".join(str(q).split())
            if q and q.lower() not in seen:
                seen.add(q.lower())
                rows.append({"qid": f"train-offtopic-{lang}-{i}-{len(rows)}", "type": "offtopic", "lang": lang,
                             "query": q, "target_id": None, "model": model, "job": job})
        append(out, rows)
        print(f"  konu dışı {i + 1}/{len(jobs)}", end="\r")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=config.DB_PATH)
    p.add_argument("--out-dir", type=Path, default=config.TRAIN_DIR)
    p.add_argument("--n-papers", type=int, default=1500)
    p.add_argument("--n-offtopic", type=int, default=1000)
    p.add_argument("--models", default=",".join(config.LLM_GEN_MODELS))
    p.add_argument("--min-interval", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("Model listesi boş. .env içinde LLM_GEN_MODELS satırını kontrol et.")
    llm._client()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    queries_out, offtopic_out = args.out_dir / "queries.jsonl", args.out_dir / "offtopic.jsonl"
    dropped_out = args.out_dir / "dropped.jsonl"
    offtopic_pool = ModelPool(models, args.min_interval)
    pool = ModelPool(models, args.min_interval)

    eval_rows = read_jsonl(config.EVAL_QUERIES_PATH)
    excluded = {r["target_id"] for r in eval_rows if r["target_id"]}
    eval_questions = {r["query"].lower() for r in eval_rows if r["type"] == "offtopic"}

    generate_offtopic(offtopic_pool, offtopic_out, args.n_offtopic, eval_questions)
    pool.inherit_account_limits(offtopic_pool)

    conn = connect(args.db)
    pool_ids = sorted(r["id"] for r in conn.execute("SELECT id FROM papers")
                      if split_of(r["id"]) == "train" and r["id"] not in excluded)
    random.Random(args.seed).shuffle(pool_ids)
    chosen = pool_ids[: args.n_papers]
    done = {r["target_id"] for r in read_jsonl(queries_out)}
    failures = 0
    dropped = Counter()
    print(f"\nEğitim havuzundan {len(chosen):,} makale seçildi, {len(done & set(chosen)):,} tanesi zaten hazır.")

    for i, pid in enumerate(chosen, start=1):
        if pid in done:
            continue
        row = conn.execute("SELECT title, abstract FROM papers WHERE id = ?", (pid,)).fetchone()
        try:
            data, model = pool.call(PAPER_PROMPT.format(title=row["title"], abstract=row["abstract"][:1200]))
        except RuntimeError as e:
            if str(e) == "too-many-attempts":
                failures += 1
                continue
            print("\nKullanılabilir model kalmadı (kota doldu ya da devre dışı). Yarın aynı komutla kaldığın yerden devam et.")
            break
        items = {key: clean_query(data.get(key), row["title"]) for key in ("en", "tr", "mixed")}
        if not items["en"]:
            failures += 1
            continue
        for key in ("tr", "mixed"):
            if not items[key]:
                continue
            reason = "" if new_words(items[key], items["en"]) else "ingilizcenin_tekrari"
            if reason:
                dropped[key] += 1
                append(dropped_out, [{"target_id": pid, "key": key, "reason": reason, "query": items[key],
                                      "en": items["en"], "model": model}])
                items[key] = ""
        rows = [{"qid": f"train-synthetic-en-{pid}", "type": "synthetic", "lang": "en", "query": items["en"],
                 "target_id": pid, "model": model}]
        if items["tr"]:
            rows.append({"qid": f"train-synthetic-tr-{pid}", "type": "synthetic", "lang": "tr", "query": items["tr"],
                         "target_id": pid, "model": model})
        if items["mixed"]:
            rows.append({"qid": f"train-mixed-tr-{pid}", "type": "mixed", "lang": "tr", "query": items["mixed"],
                         "target_id": pid, "model": model})
        append(queries_out, rows)
        print(f"  makale {i}/{len(chosen)}", end="\r")

    rows = read_jsonl(queries_out)
    per_model = defaultdict(int)
    for r in rows:
        per_model[r["model"]] += 1
    print(f"\n{queries_out}: {len(rows):,} sorgu, {len({r['target_id'] for r in rows}):,} makale")
    print(f"{offtopic_out}: {len(read_jsonl(offtopic_out)):,} konu dışı soru")
    print(f"Atlanan makale: {failures}")
    print(f"Türkçe olmadığı ya da İngilizcenin kopyası olduğu için atılan: {dict(dropped)}")
    print("Modele göre sorgu sayısı:", dict(per_model))
    print("Bu çalıştırmada harcanan token:", dict(pool.tokens + offtopic_pool.tokens))


if __name__ == "__main__":
    main()