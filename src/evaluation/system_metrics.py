"""System metrics: latency, cost per query."""
from src.schema import RetrievalResult


def avg_latency(results: list[RetrievalResult]) -> float:
    """Average latency across retrievers used."""
    if not results:
        return 0.0
    return sum(r.latency_seconds for r in results) / len(results)


def total_cost(results: list[RetrievalResult]) -> float:
    """Sum of cost across retrievers."""
    return sum(r.cost for r in results)
