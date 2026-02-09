"""Index paths and load/save helpers for built indexes."""
from pathlib import Path

def _processed_dir() -> Path:
    try:
        from config.settings import DATA_PROCESSED
        return DATA_PROCESSED
    except Exception:
        return Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"


def index_path(name: str, suffix: str) -> Path:
    """e.g. index_path('general', 'bm25.pkl') -> data/processed/general_bm25.pkl"""
    return _processed_dir() / f"{name}_{suffix}"
