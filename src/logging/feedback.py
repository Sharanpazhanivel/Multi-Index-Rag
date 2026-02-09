"""Record user feedback (thumbs up/down) or proxy reward for bandit."""
import json
from pathlib import Path
from datetime import datetime, timezone

from src.schema import RouterDecision

_DEFAULT_LOG_DIR: Path | None = None


def set_default_log_dir(path: Path) -> None:
    global _DEFAULT_LOG_DIR
    _DEFAULT_LOG_DIR = path


def record_feedback(
    query: str,
    decision: RouterDecision,
    reward: float,
    metadata: dict | None = None,
    log_dir: Path | None = None,
    query_id: str | None = None,
) -> None:
    """Store (context, action, reward) for bandit updates. Appends to feedback_log.jsonl."""
    dir_path = log_dir or _DEFAULT_LOG_DIR
    if not dir_path:
        return
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    log_file = dir_path / "feedback_log.jsonl"
    record = {
        "query": query,
        "query_id": query_id,
        "action_id": decision.action_id,
        "retriever_names": decision.retriever_names,
        "reward": reward,
        "metadata": metadata or {},
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
