"""Test evaluation metrics."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import Chunk
from src.evaluation import hit_at_k, mrr, exact_match, f1_score


def test_hit_at_k() -> None:
    chunks = [
        Chunk(text="a", score=1.0, source="x", doc_id="d1"),
        Chunk(text="b", score=0.9, source="x", doc_id="d2"),
    ]
    assert hit_at_k(chunks, {"d1"}, k=1) == 1.0
    assert hit_at_k(chunks, {"d3"}, k=5) == 0.0
    assert hit_at_k(chunks, {"d2"}, k=2) == 1.0


def test_mrr() -> None:
    chunks = [
        Chunk(text="a", score=1.0, source="x", doc_id="d1"),
        Chunk(text="b", score=0.9, source="x", doc_id="d2"),
    ]
    assert mrr(chunks, {"d2"}) == 0.5
    assert mrr(chunks, {"d1"}) == 1.0
    assert mrr(chunks, {"d3"}) == 0.0


def test_exact_match() -> None:
    assert exact_match("Yes", "yes") == 1.0
    assert exact_match("No", "Yes") == 0.0


def test_f1_score() -> None:
    assert f1_score("a b c", "a b") > 0.0
    assert f1_score("", "a") == 0.0
