import json
import re
from functools import lru_cache
from typing import Optional

from openai import OpenAI

from core import config


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY boş. .env dosyasını kontrol et.")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=6)


def chat(prompt: str, model: str, system: str = "", reasoning_effort: str = "low",
         max_retries: Optional[int] = None, max_tokens: int = 1024) -> tuple[dict, int]:
    client = _client() if max_retries is None else _client().with_options(max_retries=max_retries)
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    extra = {"reasoning_effort": reasoning_effort} if "gpt-oss" in model else {}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        **extra,
    )
    tokens = response.usage.total_tokens if response.usage else 0
    return json.loads(response.choices[0].message.content), tokens


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