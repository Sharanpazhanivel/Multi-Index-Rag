"""Test RRF merge and re-rank."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import Chunk, RetrievalResult
from src.rag.reranker import merge_and_rerank, rrf_score


def test_rrf_score() -> None:
    assert rrf_score(1) > rrf_score(2)
    assert rrf_score(1) == 1.0 / (60 + 1)


def test_merge_and_rerank_single() -> None:
    r = RetrievalResult(
        query="q",
        retriever_name="a",
        chunks=[
            Chunk(text="x", score=1.0, source="a"),
            Chunk(text="y", score=0.9, source="a"),
        ],
        latency_seconds=0.1,
    )
    out = merge_and_rerank([r], top_k=5)
    assert len(out) == 2
    assert out[0].text == "x" and out[1].text == "y"


def test_merge_and_rerank_multi() -> None:
    r1 = RetrievalResult(
        query="q",
        retriever_name="a",
        chunks=[Chunk(text="x", score=1.0, source="a"), Chunk(text="z", score=0.5, source="a")],
        latency_seconds=0.1,
    )
    r2 = RetrievalResult(
        query="q",
        retriever_name="b",
        chunks=[Chunk(text="x", score=0.9, source="b"), Chunk(text="y", score=0.8, source="b")],
        latency_seconds=0.1,
    )
    out = merge_and_rerank([r1, r2], top_k=10)
    # x appears in both -> higher RRF; then z and y
    assert len(out) == 3
    assert out[0].text == "x"  # fused score highest (rank 1 + rank 1)
