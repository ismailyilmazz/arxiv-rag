import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from scripts.deploy import wait_live

QUERY = "mixture of experts language models"
QUERIES_FILE = Path(__file__).resolve().parents[2] / "eval" / "queries.jsonl"
SAMPLE = 30
MIN_HIT = {"title": 0.9, "synthetic": 0.8}
MAX_OFFTOPIC_ACCEPTED = 0.2
SPACE_URL = "https://ismailylmz283-arxiv-rag.hf.space"


def search(url: str, query: str = QUERY, k: int = 5, timeout: int = 120):
    body = json.dumps({"query": query, "k": k}).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/search", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        return {"error": str(error)}


def sample_queries(path: Path = QUERIES_FILE, size: int = SAMPLE) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    picked = {}
    for kind in ("title", "synthetic", "offtopic"):
        group = sorted((r for r in rows if r["lang"] == "en" and r["type"] == kind), key=lambda r: r["qid"])
        step = max(1, len(group) // size)
        picked[kind] = group[::step][:size]
    return picked


def quality(url: str, queries: dict, find=search) -> tuple[dict, list[str]]:
    report, problems = {}, []
    for kind, rows in queries.items():
        hits, accepted, misses = 0, 0, []
        for row in rows:
            result = find(url, row["query"], k=10)
            if "error" in result:
                return report, [f"{kind}: arama isteği başarısız: {result['error']}"]
            accepted += result.get("accepted") is True
            if kind != "offtopic":
                found = row["target_id"] in [r["id"] for r in result.get("results") or []]
                hits += found
                if not found:
                    misses.append(row["qid"])
        n = max(1, len(rows))
        report[kind] = {"n": len(rows), "accepted": round(accepted / n, 3)}
        if kind != "offtopic":
            report[kind].update({"hit@10": round(hits / n, 3), "misses": misses})
        if kind in MIN_HIT and hits / n < MIN_HIT[kind]:
            problems.append(f"{kind}/en hit@10 {hits / n:.2f} < {MIN_HIT[kind]}; kaçanlar: {misses[:5]}")
        if kind == "offtopic" and accepted / n > MAX_OFFTOPIC_ACCEPTED:
            problems.append(f"offtopic/en kabul oranı {accepted / n:.2f} > {MAX_OFFTOPIC_ACCEPTED}")
    return report, problems


def run(
    url: str,
    wake_timeout: float = 900,
    health=wait_live.fetch_health,
    find=search,
    sleep=None,
    queries: dict | None = None,
) -> int:
    state = {}

    def ready():
        state["health"] = health(url)
        return bool(state["health"]) and state["health"].get("pipeline_loaded") is True

    kwargs = {"sleep": sleep} if sleep else {}
    if not wait_live.wait_until(ready, wake_timeout, 30, **kwargs):
        print(f"site {int(wake_timeout)} sn içinde hazır olmadı; son sağlık cevabı: {state.get('health')}")
        return 1
    data = state["health"]
    print(
        f"sağlık: commit {(data.get('build') or {}).get('commit')}, veri {data.get('data')}, gemini {data.get('gemini')}"
    )
    if data.get("status") != "ok":
        print(f"sağlık durumu ok değil: {data.get('status')}")
        return 1
    result = find(url)
    if "error" in result:
        print(f"arama isteği başarısız: {result['error']}")
        return 1
    count = len(result.get("results") or [])
    print(f"arama: kabul {result.get('accepted')}, {count} sonuç, süre {result.get('timings_ms')}")
    if result.get("accepted") is not True or count == 0:
        print("arama beklenen sonucu vermedi")
        return 1
    if queries:
        report, problems = quality(url, queries, find)
        for kind, values in report.items():
            print(f"kalite {kind}/en: {values}")
        if problems:
            print(
                "kalite eşiğin altında, modeller kendiliğinden yeniden eğitilmez; önce sebebi araştır: "
                + " | ".join(problems)
            )
            return 1
    print("duman testi geçti")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Canlı siteyi yoklar ve LLM kotası harcamayan bir arama yapar.")
    p.add_argument("--url", default=SPACE_URL)
    p.add_argument("--skip-quality", action="store_true", help="Değerlendirme setinden kalite ölçümünü atla")
    args = p.parse_args()
    sys.exit(run(args.url, queries=None if args.skip_quality else sample_queries()))


if __name__ == "__main__":
    main()
