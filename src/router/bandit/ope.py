"""Off-policy evaluation: mean reward from log; optionally estimate value of a new policy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.router.base import BaseRouter
from src.schema import RouterDecision


def load_feedback_log(log_path: Path) -> list[dict]:
    """Load feedback_log.jsonl into list of {query, action_id, reward, ...}."""
    out = []
    if not log_path.exists():
        return out
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def mean_reward_from_log(log_path: Path) -> tuple[float, int]:
    """Return (mean reward, count) from feedback log."""
    rows = load_feedback_log(log_path)
    if not rows:
        return 0.0, 0
    total = sum(r.get("reward", 0) for r in rows)
    return total / len(rows), len(rows)


def evaluate_policy_on_log(
    log_path: Path,
    policy: BaseRouter,
) -> tuple[float, int]:
    """Re-route each logged query with policy; count agreement and mean reward when action matches."""
    rows = load_feedback_log(log_path)
    if not rows:
        return 0.0, 0
    match_rewards = []
    for r in rows:
        query = r.get("query", "")
        if not query:
            continue
        decision = policy.route(query)
        if decision.action_id == r.get("action_id"):
            match_rewards.append(r.get("reward", 0))
    if not match_rewards:
        return 0.0, len(rows)
    return sum(match_rewards) / len(match_rewards), len(match_rewards)
