from dataclasses import dataclass
from pathlib import Path

MAX_BATCH_SIZE = 8
MAX_BATCH_WAIT_MS = 50
QUEUE_MAX_SIZE = 100


@dataclass
class Settings:
    database_path: Path = Path("agro_scoring.db")
    inference_timeout_seconds: float = 2.0
    llm_base_url: str = "https://vireonix.ai/v1"
    llm_model: str = "auto"
    llm_timeout_seconds: float = 60.0
