import json
from functools import lru_cache

from openai import OpenAI

from core import config


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY boş. .env dosyasını kontrol et.")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=6)


def chat_json(prompt: str, model: str, system: str = "", reasoning_effort: str = "low") -> dict:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    response = _client().chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        reasoning_effort=reasoning_effort,
    )
    return json.loads(response.choices[0].message.content)
