"""Pluggable retriever interface."""
from abc import ABC, abstractmethod

from src.schema import Chunk, RetrievalResult


class BaseRetriever(ABC):
    """Interface for all retrievers: query -> top-k chunks + latency/cost."""

    name: str

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        """Return top-k chunks with metadata, latency, and cost."""
        ...
