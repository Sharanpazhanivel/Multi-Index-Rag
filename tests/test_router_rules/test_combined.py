"""Test combined rules router (keyword first, centroid fallback)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.router.rules import CombinedRulesRouter
from src.schema import RouterDecision


def test_combined_keyword_wins_for_sql() -> None:
    router = CombinedRulesRouter(use_centroid_fallback=True)
    d = router.route("SELECT * FROM users")
    assert d.action_id == 2
    assert "structured" in d.retriever_names


def test_combined_centroid_fallback_for_default_query() -> None:
    # No centroid file -> centroid returns default action 3
    router = CombinedRulesRouter(use_centroid_fallback=True)
    d = router.route("What is the refund policy?")
    assert d.action_id == 3
    assert "technical" in d.retriever_names and "general" in d.retriever_names


def test_combined_no_centroid_fallback() -> None:
    router = CombinedRulesRouter(use_centroid_fallback=False)
    d = router.route("What is the refund policy?")
    assert d.action_id == 3
    assert d.metadata and d.metadata.get("rule") == "default"
