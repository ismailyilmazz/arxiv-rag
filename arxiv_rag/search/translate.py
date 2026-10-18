import json
import sys
from pathlib import Path
from typing import Callable, Optional

from arxiv_rag import config
from arxiv_rag.search.lang import is_turkish

PROMPT = """Translate this Turkish search query into English for searching scientific papers on arXiv.
Keep technical terms, acronyms, model and dataset names unchanged. Do not explain or add words.
Return only the English query.

Query: {query}"""


class QueryTranslator:
    def __init__(
        self,
        complete: Optional[Callable] = None,
        model: Optional[str] = None,
        cache_path: Optional[Path] = None,
        max_cache: int = 5000,
    ):
        self._complete = complete
        self.model = model or config.LLM_MODEL_FAST
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.max_cache = max_cache
        self.tokens = 0
        self.failures = 0

    def complete(self, prompt: str) -> tuple[str, int]:
        if self._complete is None:
            from arxiv_rag.generation import llm

            self._complete = llm.complete
        text, tokens, _ = self._complete(prompt, self.model, max_tokens=300, reasoning_effort="low")
        return text, tokens

    def __call__(self, text: str) -> Optional[str]:
        query = " ".join(text.split())
        if not query or not is_turkish(query):
            return None
        if query not in self.cache:
            try:
                raw, tokens = self.complete(PROMPT.format(query=query))
            except Exception as error:
                self.failures += 1
                print(f"Çeviri başarısız, orijinal sorgu kullanılıyor: {type(error).__name__}", file=sys.stderr)
                return None
            self.tokens += tokens
            line = next((s for s in raw.splitlines() if s.strip()), "").strip().strip("\"'`").strip()
            if not line or len(line) > 4 * len(query) + 40:
                return None
            if len(self.cache) >= self.max_cache:
                self.cache.pop(next(iter(self.cache)))
            self.cache[query] = line
            if self.cache_path:
                self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.cache[query]
