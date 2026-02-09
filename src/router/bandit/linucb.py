"""LinUCB contextual bandit: disjoint linear model per arm with UCB exploration."""
from __future__ import annotations

import numpy as np


class LinUCB:
    """Contextual bandit: context x -> select action; update(x, a, r)."""

    def __init__(self, dim: int, num_actions: int, alpha: float = 0.5, reg: float = 1.0) -> None:
        self.dim = dim
        self.num_actions = num_actions
        self.alpha = alpha  # exploration
        self.reg = reg
        # Per-arm: A_a (dim x dim), b_a (dim,)
        self._A = [np.eye(dim) * reg for _ in range(num_actions)]
        self._b = [np.zeros(dim) for _ in range(num_actions)]

    def _x(self, context: np.ndarray) -> np.ndarray:
        x = np.asarray(context, dtype=np.float64).ravel()
        if x.size != self.dim:
            raise ValueError(f"Context dim {x.size} != {self.dim}")
        return x.reshape(-1, 1)  # (dim, 1)

    def select(self, context: np.ndarray | list[float]) -> int:
        """Choose action with highest UCB: x' theta_a + alpha * sqrt(x' A_a^{-1} x)."""
        x = self._x(context)
        x_flat = x.ravel()
        best = -np.inf
        best_a = 0
        for a in range(self.num_actions):
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]
            mean = (x_flat @ theta).item()
            var = (x.T @ A_inv @ x).item()
            ucb = mean + self.alpha * np.sqrt(max(0, var))
            if ucb > best:
                best = ucb
                best_a = a
        return best_a

    def update(self, context: np.ndarray | list[float], action: int, reward: float) -> None:
        """Update arm's A and b with (x, r)."""
        x = self._x(context).ravel()
        self._A[action] += np.outer(x, x)
        self._b[action] += reward * x

    def get_theta(self, action: int) -> np.ndarray:
        """Return estimated theta for arm (for inspection)."""
        return np.linalg.solve(self._A[action], self._b[action])
