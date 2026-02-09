"""Simple re-ranker: merge multi-retriever results with Reciprocal Rank Fusion (RRF)."""
from src.schema import Chunk, RetrievalResult

# RRF constant (typically 60); rank is 1-based
RRF_K = 60


def rrf_score(rank: int) -> float:
    """1 / (k + rank)."""
    return 1.0 / (RRF_K + rank)


def merge_and_rerank(
    results: list[RetrievalResult],
    top_k: int = 10,
) -> list[Chunk]:
    """Merge chunks from multiple retrievers using RRF, return top_k by fused score."""
    # Aggregate by chunk identity: use (text, source, doc_id) as key to sum RRF scores
    key_to_chunk: dict[tuple[str, str, str], tuple[Chunk, float]] = {}
    for res in results:
        for rank, c in enumerate(res.chunks, start=1):
            key = (c.text, c.source, c.doc_id or "")
            score = rrf_score(rank)
            if key not in key_to_chunk:
                key_to_chunk[key] = (c, 0.0)
            prev_c, prev_score = key_to_chunk[key]
            key_to_chunk[key] = (prev_c, prev_score + score)
    # Sort by fused score descending, take top_k
    sorted_chunks = sorted(
        key_to_chunk.values(),
        key=lambda x: -x[1],
    )
    out: list[Chunk] = []
    for c, fused in sorted_chunks[:top_k]:
        out.append(Chunk(text=c.text, score=fused, source=c.source, doc_id=c.doc_id, metadata=c.metadata))
    return out
