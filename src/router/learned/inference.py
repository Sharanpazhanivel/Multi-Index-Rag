"""Load trained router from checkpoint directory."""
from pathlib import Path

import yaml

from .model import LearnedRouter

# Default path to router_actions.yaml (project root / config)
def _default_actions_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "config" / "router_actions.yaml"


def load_router(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    device: str | None = None,
) -> LearnedRouter:
    """Load LearnedRouter from a checkpoint directory (contains router_config.json, model.pt, tokenizer/)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path must be a directory: {checkpoint_path}")

    config_path = Path(config_path) if config_path else _default_actions_path()
    if config_path.exists():
        with open(config_path) as f:
            actions = yaml.safe_load(f)["actions"]
        num_actions = len(actions)
    else:
        num_actions = 5

    return LearnedRouter(num_actions=num_actions, model_path=checkpoint_path, device=device)
