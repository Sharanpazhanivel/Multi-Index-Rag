"""REINFORCE policy gradient over transformer router: sample action, get reward, update."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from src.router.base import BaseRouter
from src.schema import RouterDecision
from src.router.bandit.arms import get_action_retrievers

# Import learned router components for checkpoint loading
from src.router.learned.model import RouterModel, DEFAULT_ENCODER, CHECKPOINT_CONFIG, CHECKPOINT_MODEL, CHECKPOINT_TOKENIZER
from transformers import AutoTokenizer


class ReinforceRouter(BaseRouter):
    """Sample action from policy (softmax over router logits); update with REINFORCE on reward."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        num_actions: int = 5,
        lr: float = 1e-5,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.num_actions = num_actions
        self._encoder_name = DEFAULT_ENCODER
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: RouterModel | None = None
        self._tokenizer = None
        self.optimizer = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint()
        else:
            self._model = RouterModel(DEFAULT_ENCODER, num_actions).to(self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_ENCODER)
            self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._last: dict[str, Any] | None = None  # for update(reward)

    def _load_checkpoint(self) -> None:
        with open(self.checkpoint_path / CHECKPOINT_CONFIG) as f:
            config = json.load(f)
        self.num_actions = config["num_actions"]
        self._encoder_name = config.get("encoder_name", DEFAULT_ENCODER)
        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path / CHECKPOINT_TOKENIZER)
        self._model = RouterModel(self._encoder_name, self.num_actions).to(self._device)
        try:
            state = torch.load(self.checkpoint_path / CHECKPOINT_MODEL, map_location=self._device, weights_only=True)
        except TypeError:
            state = torch.load(self.checkpoint_path / CHECKPOINT_MODEL, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.train()
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-5)

    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        self._model.train()
        enc = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        logits = self._model(enc["input_ids"], enc["attention_mask"])
        dist = Categorical(logits=logits)
        action_id = int(dist.sample().item())
        log_prob = dist.log_prob(torch.tensor([action_id], device=self._device))
        self._last = {
            "log_prob": log_prob,
            "action_id": action_id,
        }
        return RouterDecision(
            action_id=action_id,
            retriever_names=get_action_retrievers(action_id),
            metadata={"source": "reinforce"},
        )

    def update(self, reward: float) -> None:
        """REINFORCE update: gradient = reward * grad(log_prob). Call after route() when feedback arrives."""
        if self._last is None:
            return
        self.optimizer.zero_grad()
        loss = -self._last["log_prob"] * reward
        loss.backward()
        self.optimizer.step()
        self._last = None

    def update_from_log(self, query: str, action_id: int, reward: float) -> None:
        """Off-policy style update: re-forward query, get log_prob for logged action, then REINFORCE step."""
        self._model.train()
        enc = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        logits = self._model(enc["input_ids"], enc["attention_mask"])
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(torch.tensor([action_id], device=self._device))
        self.optimizer.zero_grad()
        loss = -log_prob * reward
        loss.backward()
        self.optimizer.step()

    def save(self, output_dir: Path | str) -> None:
        """Save current policy (e.g. after online updates)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), output_dir / CHECKPOINT_MODEL)
        self._tokenizer.save_pretrained(output_dir / CHECKPOINT_TOKENIZER)
        with open(output_dir / CHECKPOINT_CONFIG, "w") as f:
            json.dump({"num_actions": self.num_actions, "encoder_name": self._encoder_name}, f)
