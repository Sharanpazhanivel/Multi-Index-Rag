"""Epsilon-greedy: with prob epsilon random action, else use base policy (e.g. learned router)."""
from __future__ import annotations

import random
from typing import Callable

from src.router.base import BaseRouter
from src.schema import RouterDecision
from .arms import get_action_retrievers


class EpsilonGreedyRouter(BaseRouter):
    """With probability epsilon pick random action; else delegate to base_router."""

    def __init__(
        self,
        base_router: BaseRouter,
        num_actions: int = 5,
        epsilon: float = 0.1,
    ) -> None:
        self.base_router = base_router
        self.num_actions = num_actions
        self.epsilon = epsilon

    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        if random.random() < self.epsilon:
            action_id = random.randint(0, self.num_actions - 1)
            return RouterDecision(
                action_id=action_id,
                retriever_names=get_action_retrievers(action_id),
                metadata={"source": "epsilon_greedy_random"},
            )
        return self.base_router.route(query, metadata)
