"""Router interface: query (+ optional metadata) -> RouterDecision."""
from abc import ABC, abstractmethod

from src.schema import RouterDecision


class BaseRouter(ABC):
    """Interface for rules-based, learned, and bandit routers."""

    @abstractmethod
    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        """Return which retrievers to use for this query."""
        ...
