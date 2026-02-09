"""Log router decision, retrievers used, latency for training/bandit."""
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from src.schema import RouterDecision

# Default log dir when none provided (use config.settings in callers)
_DEFAULT_LOG_DIR: Path | None = None


def set_default_log_dir(path: Path) -> None:
    global _DEFAULT_LOG_DIR
    _DEFAULT_LOG_DIR = path


def log_router_decision(
    query: str,
    decision: RouterDecision,
    latency_seconds: float,
    log_dir: Path | None = None,
    query_id: str | None = None,
) -> None:
    """Append one row (JSONL) for later training and off-policy eval."""
    dir_path = log_dir or _DEFAULT_LOG_DIR
    if not dir_path:
        return
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    log_file = dir_path / "router_log.jsonl"
    record = {
        "query": query,
        "query_id": query_id,
        "action_id": decision.action_id,
        "retriever_names": decision.retriever_names,
        "metadata": decision.metadata,
        "latency_seconds": round(latency_seconds, 4),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def now_seconds() -> float:
    return time.perf_counter()
