import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque

from arxiv_rag import config

RETRY_CODES = (429, 500, 502, 503, 504)
RECHECK_SECONDS = 600
_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_STATE = {"info": None, "at": 0.0}
_CALLS: deque = deque()
_USAGE = {"calls": 0, "tokens": 0, "errors": 0, "seconds": 0.0}


class GeminiUnavailable(RuntimeError):
    pass


def available() -> bool:
    return bool(config.GEMINI_API_KEY)


def usage() -> dict:
    with _LOCK:
        return {**_USAGE, "seconds": round(_USAGE["seconds"], 1)}


def reset_usage() -> None:
    with _LOCK:
        _USAGE.update(calls=0, tokens=0, errors=0, seconds=0.0)


def _trip(reason: str) -> None:
    with _STATE_LOCK:
        info = dict(_STATE["info"] or {"enabled": True})
        info.update(ok=False, errors={**(info.get("errors") or {}), "runtime": reason[:120]})
        _STATE.update(info=info, at=time.monotonic())


def forget() -> None:
    with _STATE_LOCK:
        _STATE.update(info=None, at=0.0)


def pick_model(models: list[str], wanted: str) -> str:
    if wanted in models:
        return wanted
    plain = [m for m in models if re.fullmatch(r"gemini-[\d.]+-flash", m)]
    if plain:
        return max(plain, key=lambda m: [int(x) for x in re.findall(r"\d+", m)])
    flash = [m for m in models if "flash" in m]
    return flash[0] if flash else ""


def _short(error: Exception) -> str:
    return re.sub(r"\s+", " ", str(error))[:120]


def _probe() -> dict:
    info, errors = {"enabled": True, "ok": False}, {}
    order = [config.GEMINI_AUTH] + [a for a in ("bearer", "header") if a != config.GEMINI_AUTH]
    for auth in order:
        try:
            models = list_models(auth)
        except GeminiUnavailable as e:
            errors[auth] = _short(e)
            continue
        model = pick_model(models, config.GEMINI_MODEL)
        info.update(auth=auth, model=model, flash=[m for m in models if "flash" in m][:12])
        if not model:
            errors["model"] = "listede flash modeli yok"
            break
        try:
            _, tokens, _ = complete("Reply with the single word OK.", max_tokens=200, model=model, auth=auth)
            info.update(ok=True, probe_tokens=tokens)
        except GeminiUnavailable as e:
            errors["probe"] = _short(e)
        break
    if errors:
        info["errors"] = errors
    return info


def status(force: bool = False) -> dict:
    if not available():
        return {"enabled": False}
    with _STATE_LOCK:
        info, at = _STATE["info"], _STATE["at"]
        fresh = info and (info.get("ok") or time.monotonic() - at < RECHECK_SECONDS)
        if fresh and not force:
            return dict(info)
    info = _probe()
    with _STATE_LOCK:
        _STATE.update(info=info, at=time.monotonic())
    return dict(info)


def _route(model: str, auth: str) -> tuple[str, str]:
    if model and auth:
        return model, auth
    info = status()
    return model or info.get("model") or config.GEMINI_MODEL, auth or info.get("auth") or config.GEMINI_AUTH


def headers(auth: str = "") -> dict:
    base = {"Content-Type": "application/json"}
    if (auth or config.GEMINI_AUTH) == "header":
        return {**base, "x-goog-api-key": config.GEMINI_API_KEY}
    return {**base, "Authorization": f"Bearer {config.GEMINI_API_KEY}"}


def _wait_turn() -> None:
    while config.GEMINI_RPM > 0:
        with _LOCK:
            now = time.monotonic()
            while _CALLS and now - _CALLS[0] >= 60:
                _CALLS.popleft()
            if len(_CALLS) < config.GEMINI_RPM:
                _CALLS.append(now)
                return
            pause = max(0.5, 60 - (now - _CALLS[0]))
        time.sleep(pause)


def _body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", "replace")[:300]
    except Exception:
        return ""


def _request(path: str, payload=None, auth: str = "", retries: int = 3, timeout: int = 120):
    if not available():
        raise GeminiUnavailable("GEMINI_API_KEY boş.")
    url = f"{config.GEMINI_BASE_URL}/{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    for attempt in range(retries):
        _wait_turn()
        try:
            request = urllib.request.Request(url, data=data, headers=headers(auth))
            with urllib.request.urlopen(request, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in RETRY_CODES and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            with _LOCK:
                _USAGE["errors"] += 1
            if e.code in RETRY_CODES:
                _trip(f"HTTP {e.code}")
            raise GeminiUnavailable(f"HTTP {e.code}: {_body(e)}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            with _LOCK:
                _USAGE["errors"] += 1
            _trip(type(e).__name__)
            raise GeminiUnavailable(f"{type(e).__name__}: {e}") from e
    raise GeminiUnavailable("Gemini cevap vermedi.")


def list_models(auth: str = "") -> list[str]:
    data = _request("models", auth=auth) or {}
    return sorted(str(m.get("id", "")).removeprefix("models/") for m in data.get("data") or [] if m.get("id"))


def _send(payload: dict, auth: str) -> tuple[dict, int]:
    start = time.monotonic()
    try:
        data = _request("chat/completions", payload, auth=auth) or {}
    finally:
        with _LOCK:
            _USAGE["seconds"] += time.monotonic() - start
    tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
    with _LOCK:
        _USAGE["calls"] += 1
        _USAGE["tokens"] += tokens
    choices = data.get("choices") or []
    if not choices:
        raise GeminiUnavailable("Gemini boş cevap döndürdü.")
    return choices[0], tokens


def complete(prompt: str, max_tokens: int = 2000, model: str = "", auth: str = "") -> tuple[str, int, str]:
    model, auth = _route(model, auth)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    choice, tokens = _send(payload, auth)
    return ((choice.get("message") or {}).get("content") or "").strip(), tokens, choice.get("finish_reason") or ""


def chat(prompt: str, max_tokens: int = 2000, model: str = "", auth: str = "") -> tuple[dict, int]:
    model, auth = _route(model, auth)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    choice, tokens = _send(payload, auth)
    text = ((choice.get("message") or {}).get("content") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text), tokens
    except json.JSONDecodeError as e:
        raise GeminiUnavailable(f"Gemini geçersiz JSON döndürdü: {text[:120]}") from e
