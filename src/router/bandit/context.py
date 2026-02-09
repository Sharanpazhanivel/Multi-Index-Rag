"""Context for contextual bandit: query embedding + cost/latency preferences."""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel


class BanditContext(BaseModel):
    """Input to bandit: what we observe before choosing an action."""
    query_embedding: list[float] | None = None
    query_text: str = ""
    cost_preference: float = 0.0  # weight on cost in reward
    latency_preference: float = 0.0

    def to_vector(self, dim: int, embed_fn: Callable[[str], list[float]] | None = None) -> list[float]:
        """Build fixed-dim context vector: [embedding; cost_pref; latency_pref] or zeros if no embed_fn."""
        if embed_fn and self.query_text:
            emb = embed_fn(self.query_text)
            # Pad or truncate to dim - 2 so we can append two floats
            d = dim - 2
            if len(emb) >= d:
                vec = list(emb[:d])
            else:
                vec = list(emb) + [0.0] * (d - len(emb))
        else:
            vec = [0.0] * (dim - 2)
        vec.append(self.cost_preference)
        vec.append(self.latency_preference)
        return vec
