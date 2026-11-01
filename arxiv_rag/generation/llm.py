import json
import re
import threading
import time
from collections import defaultdict, deque
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from arxiv_rag import config

_WINDOW: dict[str, deque] = defaultdict(deque)
_WINDOW_LOCK = threading.Lock()


def reserve(model: str, tokens: int) -> float:
    limit = config.LLM_TPM
    if limit <= 0:
        return 0.0
    tokens, waited = min(tokens, limit), 0.0
    while True:
        with _WINDOW_LOCK:
            now = time.monotonic()
            window = _WINDOW[model]
            while window and now - window[0][0] >= 60:
                window.popleft()
            if sum(t for _, t in window) + tokens <= limit:
                return waited
            pause = max(0.5, 60 - (now - window[0][0]))
        time.sleep(pause)
        waited += pause


def record(model: str, tokens: int) -> None:
    with _WINDOW_LOCK:
        _WINDOW[model].append((time.monotonic(), tokens))


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY boş. .env dosyasını kontrol et.")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=6)


def chat(
    prompt: str,
    model: str,
    system: str = "",
    reasoning_effort: str = "low",
    max_retries: Optional[int] = None,
    max_tokens: int = 1024,
) -> tuple[dict, int]:
    client = _client() if max_retries is None else _client().with_options(max_retries=max_retries)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    reserve(model, (len(prompt) + len(system)) // 3 + max_tokens)
    extra = {"reasoning_effort": reasoning_effort} if "gpt-oss" in model else {}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        **extra,
    )
    tokens = response.usage.total_tokens if response.usage else 0
    record(model, tokens)
    return json.loads(response.choices[0].message.content), tokens


def request_overflow(error: Exception) -> Optional[int]:
    text = str(error)
    match = re.search(r"Limit (\d+), Requested (\d+)", text)
    if match and "too large" in text.lower():
        return int(match.group(2)) - int(match.group(1))
    return None


def complete(prompt: str, model: str, max_tokens: int = 4000, reasoning_effort: str = "low") -> tuple[str, int, str]:
    extra = {"reasoning_effort": reasoning_effort} if "gpt-oss" in model else {}
    for attempt in range(2):
        reserve(model, len(prompt) // 3 + max_tokens)
        try:
            response = _client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                **extra,
            )
            break
        except Exception as e:
            overflow = request_overflow(e)
            if attempt == 0 and overflow:
                max_tokens = max(500, max_tokens - overflow - 200)
                continue
            raise
    choice = response.choices[0]
    tokens = response.usage.total_tokens if response.usage else 0
    record(model, tokens)
    return (choice.message.content or "").strip(), tokens, choice.finish_reason or ""


def chat_json(prompt: str, model: str, system: str = "", reasoning_effort: str = "low") -> dict:
    return chat(prompt, model, system, reasoning_effort)[0]


def limit_kind(error: Exception) -> Optional[str]:
    text = str(error).lower()
    if "rate limit" not in text and "ratelimit" not in type(error).__name__.lower():
        return None
    if "per day" in text or "(tpd)" in text or "(rpd)" in text:
        return "day"
    return "minute"


def output_limit(error: Exception) -> Optional[int]:
    match = re.search(r"\(OTPM\): Limit (\d+)", str(error))
    return int(match.group(1)) if match else None


TOKEN_BUDGET = 7600
CONTINUE = (
    "\n\nThe section you are writing currently ends with:\n{tail}\n\nContinue exactly from where it stops. "
    "Do not repeat earlier text, do not start a new table and do not repeat headings. Finish the section "
    "with complete sentences."
)


def complete_continued(prompt: str, model: str, max_tokens: int) -> tuple[str, int, bool]:
    budget = max(400, min(max_tokens, TOKEN_BUDGET - len(prompt) // 3))
    text, tokens, finish = complete(prompt, model=model, max_tokens=budget)
    if finish == "length" and text.strip():
        more_prompt = prompt + CONTINUE.format(tail=text[-1200:])
        budget = max(400, min(max_tokens, TOKEN_BUDGET - len(more_prompt) // 3))
        more, extra, finish = complete(more_prompt, model=model, max_tokens=budget)
        text, tokens = text.rstrip() + " " + more.lstrip(), tokens + extra
    return text, tokens, finish == "length"
