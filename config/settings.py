"""Load settings from env and config files."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Always load .env from project root so OPENAI_API_KEY is found no matter where you run from
load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"), encoding="utf-8")
DATA_RAW = Path(os.getenv("DATA_RAW", str(PROJECT_ROOT / "data" / "raw")))
DATA_PROCESSED = Path(os.getenv("DATA_PROCESSED", str(PROJECT_ROOT / "data" / "processed")))
DATA_LABELS = Path(os.getenv("DATA_LABELS", str(PROJECT_ROOT / "data" / "labels")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "data" / "processed" / "logs")))
