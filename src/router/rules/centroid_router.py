"""Route by embedding similarity to domain centroids."""
import json
from pathlib import Path
from typing import Callable

from src.router.bandit.arms import get_action_retrievers
from src.schema import RouterDecision
from src.router.base import BaseRouter


def _cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _default_action_id() -> int:
    """Default when centroids or embeddings unavailable: vector + BM25."""
    return 3


class CentroidRouter(BaseRouter):
    """Compare query embedding to precomputed domain centroids; pick nearest action."""

    def __init__(
        self,
        centroid_path: Path | str | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        default_action_id: int | None = None,
    ) -> None:
        self.centroid_path = Path(centroid_path) if centroid_path else None
        self.embed_fn = embed_fn
        self.default_action_id = default_action_id if default_action_id is not None else _default_action_id()
        self._centroids: list[dict] = []
        self._dim: int = 0
        if self.centroid_path and self.centroid_path.exists():
            self._load_centroids()

    def _load_centroids(self) -> None:
        with open(self.centroid_path) as f:
            data = json.load(f)
        self._dim = data.get("dim", 0)
        self._centroids = data.get("centroids", [])

    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        if not self._centroids or not self.embed_fn:
            return RouterDecision(
                action_id=self.default_action_id,
                retriever_names=get_action_retrievers(self.default_action_id),
                metadata={"source": "centroid_fallback"},
            )
        vec = self.embed_fn(query)
        if len(vec) != self._dim or not vec:
            return RouterDecision(
                action_id=self.default_action_id,
                retriever_names=get_action_retrievers(self.default_action_id),
                metadata={"source": "centroid_bad_embed"},
            )
        best_action_id = self._centroids[0]["action_id"]
        best_sim = -2.0
        for c in self._centroids:
            sim = _cosine_sim(vec, c["vector"])
            if sim > best_sim:
                best_sim = sim
                best_action_id = c["action_id"]
        return RouterDecision(
            action_id=best_action_id,
            retriever_names=get_action_retrievers(best_action_id),
            metadata={"source": "centroid", "similarity": best_sim},
        )
