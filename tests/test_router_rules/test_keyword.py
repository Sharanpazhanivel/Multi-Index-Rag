"""Test keyword router."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.router.rules import KeywordRouter
from src.schema import RouterDecision


def test_keyword_router_sql(sample_sql_query: str) -> None:
    router = KeywordRouter()
    d = router.route(sample_sql_query)
    assert isinstance(d, RouterDecision)
    assert d.action_id == 2
    assert "structured" in d.retriever_names


def test_keyword_router_general(sample_query: str) -> None:
    router = KeywordRouter()
    d = router.route(sample_query)
    assert d.action_id == 3
    assert "technical" in d.retriever_names and "general" in d.retriever_names
    assert d.metadata and d.metadata.get("rule") == "default"


def test_keyword_router_metadata_sql(sample_sql_query: str) -> None:
    router = KeywordRouter()
    d = router.route(sample_sql_query)
    assert d.metadata and d.metadata.get("rule") == "sql"
