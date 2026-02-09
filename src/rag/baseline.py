"""Baseline: query all retrievers, concatenate context."""
from src.retrievers import get_retriever
from src.schema import RetrievalResult


def baseline_retrieve(
    query: str,
    retriever_names: list[str],
    top_k_per: int = 10,
) -> list[RetrievalResult]:
    """Call each retriever and return all results (for hybrid baseline)."""
    results = []
    for name in retriever_names:
        retriever = get_retriever(name)
        results.append(retriever.retrieve(query, top_k=top_k_per))
    return results
