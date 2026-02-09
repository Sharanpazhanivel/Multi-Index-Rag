"""Small transformer encoder + classification head for routing."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.schema import RouterDecision
from src.router.base import BaseRouter
from src.router.bandit.arms import get_action_retrievers

DEFAULT_ENCODER = "distilbert-base-uncased"
CHECKPOINT_CONFIG = "router_config.json"
CHECKPOINT_MODEL = "model.pt"
CHECKPOINT_TOKENIZER = "tokenizer"


class RouterModel(nn.Module):
    """Encoder + linear classification head for action prediction."""

    def __init__(self, encoder_name: str, num_actions: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_actions)
        self.num_actions = num_actions

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # [CLS]
        return self.classifier(self.dropout(pooled))


class LearnedRouter(BaseRouter):
    """Query -> tokenize -> encoder -> argmax -> action_id -> RouterDecision."""

    def __init__(
        self,
        num_actions: int = 5,
        model_path: str | Path | None = None,
        encoder_name: str = DEFAULT_ENCODER,
        device: str | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.model_path = Path(model_path) if model_path else None
        self.encoder_name = encoder_name
        self._device = device
        self._model: RouterModel | None = None
        self._tokenizer = None
        if self.model_path and self.model_path.exists():
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        config_path = self.model_path / CHECKPOINT_CONFIG
        with open(config_path) as f:
            config = json.load(f)
        self.num_actions = config["num_actions"]
        self.encoder_name = config.get("encoder_name", DEFAULT_ENCODER)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path / CHECKPOINT_TOKENIZER)
        self._model = RouterModel(self.encoder_name, self.num_actions)
        try:
            state = torch.load(self.model_path / CHECKPOINT_MODEL, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(self.model_path / CHECKPOINT_MODEL, map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        if self._model is None or self._tokenizer is None:
            aid = 3
            return RouterDecision(action_id=aid, retriever_names=get_action_retrievers(aid), metadata={"source": "learned_fallback"})
        enc = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self._model(enc["input_ids"], enc["attention_mask"])
            action_id = int(logits.argmax(dim=-1).item())
        return RouterDecision(
            action_id=action_id,
            retriever_names=get_action_retrievers(action_id),
            metadata={"source": "learned"},
        )
