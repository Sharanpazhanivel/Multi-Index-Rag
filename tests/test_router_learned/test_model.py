"""Test learned router (no checkpoint -> fallback; with checkpoint -> inference)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.router.learned import LearnedRouter, load_router
from src.schema import RouterDecision


def test_learned_router_fallback_when_no_checkpoint() -> None:
    router = LearnedRouter(num_actions=5)
    d = router.route("test query")
    assert isinstance(d, RouterDecision)
    assert d.action_id == 3  # default fallback
    assert d.metadata and d.metadata.get("source") == "learned_fallback"
    assert "technical" in d.retriever_names and "general" in d.retriever_names


def test_load_router_missing_dir_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_router(ROOT / "nonexistent_checkpoint_dir")
