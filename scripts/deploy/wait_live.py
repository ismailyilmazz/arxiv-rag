import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def fetch_health(url: str, timeout: int = 60):
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/health", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def wait_until(check, timeout: float, interval: float = 30.0, clock=time.monotonic, sleep=time.sleep) -> bool:
    deadline = clock() + timeout
    while True:
        if check():
            return True
        if clock() >= deadline:
            return False
        sleep(interval)


def idle(url: str, health=fetch_health):
    data = health(url)
    if data is None:
        return True, "site cevap vermiyor (uykuda ya da açılıyor), kuyruk boş sayıldı"
    queue = data.get("queue") or {}
    busy = {k: v for k, v in queue.items() if v}
    return not busy, f"kuyruk: {queue}"


def live(url: str, commit: str, health=fetch_health):
    data = health(url)
    if data is None:
        return False, "site henüz cevap vermiyor"
    seen = (data.get("build") or {}).get("commit")
    ready = data.get("pipeline_loaded") is True
    return seen == commit and ready, f"canlı commit {seen}, arama akışı yüklü: {ready}"


def run(
    mode: str, url: str, commit: str, timeout: float, interval: float, health=fetch_health, sleep=time.sleep
) -> int:
    last = {"note": ""}

    def check():
        ok, note = idle(url, health) if mode == "idle" else live(url, commit, health)
        if note != last["note"]:
            print(note, flush=True)
            last["note"] = note
        return ok

    if wait_until(check, timeout, interval, sleep=sleep):
        print("tamam" if mode == "idle" else f"canlıda {commit}", flush=True)
        return 0
    if mode == "idle":
        print(f"kuyruk {int(timeout)} sn içinde boşalmadı; yükleme ertelendi, yarım kalan iş olmasın diye", flush=True)
    else:
        print(f"{int(timeout)} sn içinde canlıda {commit} görülmedi; son durum: {last['note']}", flush=True)
    return 1


def main() -> None:
    p = argparse.ArgumentParser(description="Canlı Space'i yoklar: kuyruk boş mu, yeni sürüm yayında mı.")
    p.add_argument("mode", choices=["idle", "live"])
    p.add_argument("--url", required=True)
    p.add_argument("--commit", default="")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--interval", type=float, default=30)
    args = p.parse_args()
    if args.mode == "live" and not args.commit:
        p.error("live için --commit gerekli")
    sys.exit(run(args.mode, args.url, args.commit, args.timeout, args.interval))


if __name__ == "__main__":
    main()
