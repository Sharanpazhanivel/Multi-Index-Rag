"""Test retriever interface and registry."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.retrievers import get_retriever
from src.schema import RetrievalResult


def test_get_retriever_returns_result() -> None:
    r = get_retriever("general")
    out = r.retrieve("test query", top_k=3)
    assert isinstance(out, RetrievalResult)
    assert out.retriever_name == "general"
    assert out.query == "test query"
