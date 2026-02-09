"""Unified bandit router: LinUCB, epsilon-greedy, or REINFORCE; implements BaseRouter."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from src.router.base import BaseRouter
from src.schema import RouterDecision
from .arms import get_action_retrievers
from .context import BanditContext
from .linucb import LinUCB
from .epsilon_greedy import EpsilonGreedyRouter
from .reinforce import ReinforceRouter


def build_context(
    query: str,
    embed_fn: Callable[[str], list[float]] | None = None,
    cost_preference: float = 0.0,
    latency_preference: float = 0.0,
) -> BanditContext:
    return BanditContext(
        query_embedding=embed_fn(query) if embed_fn else None,
        query_text=query,
        cost_preference=cost_preference,
        latency_preference=latency_preference,
    )


class BanditRouter(BaseRouter):
    """Single router that delegates to LinUCB, epsilon-greedy, or REINFORCE."""

    def __init__(
        self,
        strategy: str = "linucb",
        num_actions: int = 5,
        *,
        # LinUCB
        context_dim: int = 386,  # 384 embed + 2 prefs
        alpha: float = 0.5,
        embed_fn: Callable[[str], list[float]] | None = None,
        # Epsilon-greedy
        base_router: BaseRouter | None = None,
        epsilon: float = 0.1,
        # REINFORCE
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.strategy = strategy
        self.num_actions = num_actions
        self.embed_fn = embed_fn
        self._context_dim = context_dim

        if strategy == "linucb":
            self._bandit = LinUCB(dim=context_dim, num_actions=num_actions, alpha=alpha)
            self._reinforce = None
            self._epsilon_router = None
        elif strategy == "epsilon_greedy":
            if base_router is None:
                from src.router.learned import LearnedRouter
                base_router = LearnedRouter(num_actions=num_actions)
            self._epsilon_router = EpsilonGreedyRouter(base_router, num_actions=num_actions, epsilon=epsilon)
            self._bandit = None
            self._reinforce = None
        elif strategy == "reinforce":
            self._reinforce = ReinforceRouter(checkpoint_path=checkpoint_path, num_actions=num_actions)
            self._bandit = None
            self._epsilon_router = None
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        if self.strategy == "linucb":
            ctx = build_context(
                query,
                embed_fn=self.embed_fn,
                cost_preference=(metadata or {}).get("cost_preference", 0.0),
                latency_preference=(metadata or {}).get("latency_preference", 0.0),
            )
            vec = ctx.to_vector(self._context_dim, self.embed_fn)
            action_id = self._bandit.select(vec)
            return RouterDecision(
                action_id=action_id,
                retriever_names=get_action_retrievers(action_id),
                metadata={"source": "linucb"},
            )
        if self.strategy == "epsilon_greedy":
            return self._epsilon_router.route(query, metadata)
        if self.strategy == "reinforce":
            return self._reinforce.route(query, metadata)
        raise RuntimeError("Unreachable")

    def update(self, query: str, decision: RouterDecision, reward: float, metadata: dict | None = None) -> None:
        """Update bandit with (context, action, reward). REINFORCE uses reward only (call after route())."""
        if self.strategy == "linucb":
            ctx = build_context(
                query,
                embed_fn=self.embed_fn,
                cost_preference=(metadata or {}).get("cost_preference", 0.0),
                latency_preference=(metadata or {}).get("latency_preference", 0.0),
            )
            vec = ctx.to_vector(self._context_dim, self.embed_fn)
            self._bandit.update(vec, decision.action_id, reward)
        elif self.strategy == "reinforce":
            self._reinforce.update(reward)

    def update_from_log_entry(self, query: str, action_id: int, reward: float, metadata: dict | None = None) -> None:
        """Update from a logged (query, action_id, reward). Use in replay; REINFORCE re-forwards to get log_prob."""
        if self.strategy == "linucb":
            ctx = build_context(
                query,
                embed_fn=self.embed_fn,
                cost_preference=(metadata or {}).get("cost_preference", 0.0),
                latency_preference=(metadata or {}).get("latency_preference", 0.0),
            )
            vec = ctx.to_vector(self._context_dim, self.embed_fn)
            self._bandit.update(vec, action_id, reward)
        elif self.strategy == "reinforce":
            self._reinforce.update_from_log(query, action_id, reward)

    def save(self, output_dir: Path | str) -> None:
        """Persist policy (REINFORCE only; LinUCB state could be saved separately)."""
        if self.strategy == "reinforce" and self._reinforce:
            self._reinforce.save(output_dir)
