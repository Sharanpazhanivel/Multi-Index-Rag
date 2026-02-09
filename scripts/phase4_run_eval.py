"""Phase 4: Run evaluation and ablations.
Usage:
  Create eval set data/labels/eval_set.jsonl with one JSON per line:
    {"query": "...", "query_id": "1", "reference_answer": "...", "gold_doc_ids": ["id1", "id2"]}
  reference_answer and gold_doc_ids are optional.
  Run: python scripts/phase4_run_eval.py
  Options: --eval-set, --checkpoint, --strategies, --output (table to file), --format (table|csv|json)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_LABELS
from src.evaluation import run_ablations, format_table, load_eval_set


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4: evaluation and ablations")
    ap.add_argument("--eval-set", type=Path, default=DATA_LABELS / "eval_set.jsonl", help="Path to eval set JSONL")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "router", help="Checkpoint dir for learned/reinforce")
    ap.add_argument("--strategies", nargs="+", default=None, help="Strategies: baseline rules learned bandit_linucb bandit_epsilon bandit_reinforce")
    ap.add_argument("--output", type=Path, default=None, help="Write table to file")
    ap.add_argument("--format", choices=["table", "csv", "json"], default="table")
    ap.add_argument("--num-actions", type=int, default=5)
    args = ap.parse_args()

    eval_set = load_eval_set(args.eval_set)
    if not eval_set:
        # Create sample eval set
        args.eval_set.parent.mkdir(parents=True, exist_ok=True)
        sample = [
            {"query": "What is the refund policy?", "query_id": "1", "reference_answer": "", "gold_doc_ids": []},
            {"query": "SELECT * FROM orders", "query_id": "2", "reference_answer": "", "gold_doc_ids": []},
        ]
        with open(args.eval_set, "w") as f:
            for s in sample:
                f.write(json.dumps(s) + "\n")
        eval_set = load_eval_set(args.eval_set)
        print(f"Created sample {args.eval_set}. Re-run to see metrics.")
        if not eval_set:
            sys.exit(1)

    rows = run_ablations(
        eval_set_path=args.eval_set,
        strategies=args.strategies,
        checkpoint_path=args.checkpoint,
        num_actions=args.num_actions,
    )

    if args.format == "table":
        out = format_table(rows)
        print(out)
    elif args.format == "csv":
        if not rows:
            out = "strategy\n"
        else:
            keys = ["strategy"] + list(rows[0].get("metrics", {}).keys()) if rows and not rows[0].get("skipped") else ["strategy"]
            out = ",".join(keys) + "\n"
            for r in rows:
                cells = [r["strategy"]]
                for k in keys[1:]:
                    v = r.get("metrics", {}).get(k, "")
                    cells.append(str(v))
                out += ",".join(cells) + "\n"
        print(out)
    else:
        out = json.dumps([{"strategy": r["strategy"], "metrics": r.get("metrics", {}), "skipped": r.get("skipped", False)} for r in rows], indent=2)
        print(out)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            if args.format == "table":
                f.write(format_table(rows))
            elif args.format == "csv":
                if not rows:
                    f.write("strategy\n")
                else:
                    keys = ["strategy"] + list(rows[0].get("metrics", {}).keys()) if rows and not rows[0].get("skipped") else ["strategy"]
                    f.write(",".join(keys) + "\n")
                    for r in rows:
                        cells = [r["strategy"]] + [str(r.get("metrics", {}).get(k, "")) for k in keys[1:]]
                        f.write(",".join(cells) + "\n")
            else:
                f.write(out)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
