"""Keyword / regex rules (e.g. SELECT, JOIN -> SQL; function/class -> code)."""
import re

from src.router.bandit.arms import get_action_retrievers
from src.schema import RouterDecision
from src.router.base import BaseRouter


# Patterns: (pattern, action_id). Order matters (first match wins).
KEYWORD_RULES = [
    (re.compile(r"\b(SELECT|JOIN|WHERE|FROM|INSERT|UPDATE|DELETE)\b", re.I), 2),   # SQL
    (re.compile(r"\b(def\s+\w+|class\s+\w+|import\s+\w+|function\s*\(|\.py\b)"), 4),  # code
]


class KeywordRouter(BaseRouter):
    """Hand-crafted keywords to pick retriever(s). Single-index (SQL, code) or multi (default)."""

    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        for pattern, action_id in KEYWORD_RULES:
            if pattern.search(query):
                return RouterDecision(
                    action_id=action_id,
                    retriever_names=get_action_retrievers(action_id),
                    metadata={"source": "keyword", "rule": "sql" if action_id == 2 else "code"},
                )
        # Default: vector + BM25 (multi-index)
        return RouterDecision(
            action_id=3,
            retriever_names=get_action_retrievers(3),
            metadata={"source": "keyword", "rule": "default"},
        )
