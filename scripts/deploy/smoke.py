import argparse
import json
import sys
import urllib.error
import urllib.request

from scripts.deploy import wait_live

QUERY = "mixture of experts language models"
SPACE_URL = "https://ismailylmz283-arxiv-rag.hf.space"


def search(url: str, query: str = QUERY, timeout: int = 120):
    body = json.dumps({"query": query, "k": 5}).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/search", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        return {"error": str(error)}


def run(url: str, wake_timeout: float = 900, health=wait_live.fetch_health, find=search, sleep=None) -> int:
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
    print("duman testi geçti")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Canlı siteyi yoklar ve LLM kotası harcamayan bir arama yapar.")
    p.add_argument("--url", default=SPACE_URL)
    args = p.parse_args()
    sys.exit(run(args.url))


if __name__ == "__main__":
    main()
