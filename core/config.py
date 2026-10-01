import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

_data_dir = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR = _data_dir if _data_dir.is_absolute() else ROOT / _data_dir
DB_PATH = DATA_DIR / "papers.db"
RESULTS_DIR = ROOT / "eval" / "results"

EVAL_QUERIES_PATH = ROOT / "eval" / "queries.jsonl"

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MODEL_FAST = os.getenv("LLM_MODEL_FAST", "openai/gpt-oss-20b")
