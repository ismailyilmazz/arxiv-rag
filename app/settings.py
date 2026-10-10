import os
from dataclasses import dataclass
from pathlib import Path

from core import config


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


@dataclass
class Settings:
    db_path: Path = config.DB_PATH
    cache_path: Path = config.CACHE_DB_PATH
    app_db_path: Path = config.DATA_DIR / "app.db"
    vectors_dir: Path = config.DATA_DIR / "vectors_dev"
    models_dir: Path = config.DATA_DIR / "models"
    output_dir: Path = config.DATA_DIR / "jobs"
    n_sources: int = 30
    plans_per_ip_hour: int = 10
    generations_per_ip_day: int = 2
    generations_per_day: int = 5
    trust_proxy: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=_path("APP_PAPERS_DB", config.DB_PATH),
            cache_path=_path("APP_CACHE_DB", config.CACHE_DB_PATH),
            app_db_path=_path("APP_DB", config.DATA_DIR / "app.db"),
            vectors_dir=_path("APP_VECTORS_DIR", config.DATA_DIR / "vectors_dev"),
            models_dir=_path("APP_MODELS_DIR", config.DATA_DIR / "models"),
            output_dir=_path("APP_OUTPUT_DIR", config.DATA_DIR / "jobs"),
            n_sources=_int("APP_N_SOURCES", 30),
            plans_per_ip_hour=_int("APP_PLANS_PER_IP_HOUR", 10),
            generations_per_ip_day=_int("APP_GENERATIONS_PER_IP_DAY", 2),
            generations_per_day=_int("APP_GENERATIONS_PER_DAY", 5),
            trust_proxy=os.getenv("APP_TRUST_PROXY", "0") == "1",
        )
