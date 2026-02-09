"""Retrieval metrics: Hit@k, MRR."""
from src.schema import Chunk


def hit_at_k(retrieved: list[Chunk], gold_doc_ids: set[str], k: int = 5) -> float:
    """1.0 if any of top-k chunks is in gold set."""
    top_ids = {c.doc_id for c in (retrieved[:k] or []) if c.doc_id}
    return 1.0 if top_ids & gold_doc_ids else 0.0


def mrr(retrieved: list[Chunk], gold_doc_ids: set[str]) -> float:
    """Mean reciprocal rank of first hit."""
    for i, c in enumerate(retrieved):
        if c.doc_id and c.doc_id in gold_doc_ids:
            return 1.0 / (i + 1)
    return 0.0
