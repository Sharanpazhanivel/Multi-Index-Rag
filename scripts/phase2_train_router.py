"""Phase 2: Train supervised transformer router on collected labels.
Usage:
  Ensure data/labels/routing_labels.jsonl exists (from phase1_collect_labels.py --offline-eval).
  Run: python scripts/phase2_train_router.py
  Options: --labels, --output, --encoder, --batch-size, --epochs, --lr
  Output: checkpoint dir with model.pt, tokenizer/, router_config.json
"""
import argparse
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_LABELS, DATA_PROCESSED
from src.router.learned import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train supervised transformer router")
    ap.add_argument("--labels", type=Path, default=DATA_LABELS / "routing_labels.jsonl", help="Path to routing_labels.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "router", help="Checkpoint output directory")
    ap.add_argument("--encoder", type=str, default="distilbert-base-uncased", help="HuggingFace encoder name")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    if not args.labels.exists():
        print(f"Labels not found: {args.labels}")
        print("Run: python scripts/phase1_collect_labels.py --offline-eval")
        sys.exit(1)

    actions_path = ROOT / "config" / "router_actions.yaml"
    num_actions = 5
    if actions_path.exists():
        with open(actions_path) as f:
            num_actions = len(yaml.safe_load(f)["actions"])

    train(
        labels_path=args.labels,
        output_dir=args.output,
        encoder_name=args.encoder,
        num_actions=num_actions,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
    )
    print("Done. Load with: load_router(checkpoint_path)")


if __name__ == "__main__":
    main()
