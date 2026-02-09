"""Map action_id -> retriever names (from config)."""
import yaml
from pathlib import Path


def get_action_retrievers(action_id: int) -> list[str]:
    config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "router_actions.yaml"
    with open(config_path) as f:
        actions = yaml.safe_load(f)["actions"]
    for a in actions:
        if a["id"] == action_id:
            return a["retrievers"]
    return []
