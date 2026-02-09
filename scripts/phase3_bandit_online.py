"""Phase 3: Bandit online loop or replay from feedback log.
Usage:
  Online (route + collect feedback, then update):
    python scripts/phase3_bandit_online.py --strategy linucb --mode online --queries data/labels/queries.jsonl
  Replay (update bandit from existing feedback_log.jsonl):
    python scripts/phase3_bandit_online.py --strategy linucb --mode replay --feedback data/processed/logs/feedback_log.jsonl
  REINFORCE replay: updates policy from log (re-computes log_prob for logged actions).
  Optional: --embed (use sentence-transformers for LinUCB context). --save for REINFORCE checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_LABELS, DATA_PROCESSED, LOG_DIR
from src.router.bandit import BanditRouter, load_feedback_log, mean_reward_from_log
from src.rag import RAGPipeline
from src.logging.feedback import record_feedback


def get_embed_fn():
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda q: m.encode(q, convert_to_numpy=True).tolist()
    except ImportError:
        return None


def run_online(
    strategy: str,
    queries_path: Path,
    log_dir: Path,
    num_actions: int = 5,
    checkpoint_path: Path | None = None,
) -> None:
    """Route each query, run RAG, log decision; expect feedback elsewhere (e.g. manual or from file)."""
    embed_fn = get_embed_fn() if strategy == "linucb" else None
    router = BanditRouter(
        strategy=strategy,
        num_actions=num_actions,
        embed_fn=embed_fn,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )
    pipeline = RAGPipeline(router=router, log_dir=log_dir)

    queries = []
    if queries_path.suffix == ".jsonl":
        with open(queries_path) as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line))
    else:
        with open(queries_path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    queries.append({"query": line.strip(), "id": str(i)})

    for q in queries:
        query = q.get("query", q.get("q", ""))
        qid = q.get("id", q.get("query_id"))
        out = pipeline.run(query=query, query_id=qid)
        decision = out["decision"]
        print(f"query_id={qid} action_id={decision.action_id} retrievers={decision.retriever_names}")
        # In a real app, feedback would come from user; here we don't block for it.
    print("Online run done. Record feedback to feedback_log.jsonl then run --mode replay to update.")


def run_replay(
    strategy: str,
    feedback_path: Path,
    log_dir: Path,
    num_actions: int = 5,
    checkpoint_path: Path | None = None,
    save_path: Path | None = None,
) -> None:
    """Update bandit from feedback log. LinUCB: update (context, action, reward). REINFORCE: re-forward and update."""
    rows = load_feedback_log(feedback_path)
    if not rows:
        print("No rows in feedback log.")
        return

    embed_fn = get_embed_fn() if strategy == "linucb" else None
    router = BanditRouter(
        strategy=strategy,
        num_actions=num_actions,
        embed_fn=embed_fn,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
    )

    for r in rows:
        query = r.get("query", "")
        action_id = r.get("action_id", 0)
        reward = float(r.get("reward", 0))
        if not query:
            continue
        router.update_from_log_entry(query, action_id, reward, metadata=r)

    if save_path and strategy == "reinforce":
        router.save(save_path)
        print(f"Saved REINFORCE checkpoint to {save_path}")

    mean, n = mean_reward_from_log(feedback_path)
    print(f"Replay done. Processed {len(rows)} rows. Mean reward in log: {mean:.4f} (n={n})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: bandit online or replay")
    ap.add_argument("--strategy", choices=["linucb", "epsilon_greedy", "reinforce"], default="linucb")
    ap.add_argument("--mode", choices=["online", "replay"], default="replay")
    ap.add_argument("--queries", type=Path, default=DATA_LABELS / "queries.jsonl")
    ap.add_argument("--feedback", type=Path, default=None)
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "router")
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--num-actions", type=int, default=5)
    args = ap.parse_args()

    log_dir = args.log_dir or LOG_DIR or (DATA_PROCESSED / "logs")
    feedback_path = args.feedback or (log_dir / "feedback_log.jsonl")

    if args.mode == "online":
        run_online(
            strategy=args.strategy,
            queries_path=args.queries,
            log_dir=log_dir,
            num_actions=args.num_actions,
            checkpoint_path=args.checkpoint if args.strategy == "reinforce" else None,
        )
    else:
        run_replay(
            strategy=args.strategy,
            feedback_path=feedback_path,
            log_dir=log_dir,
            num_actions=args.num_actions,
            checkpoint_path=args.checkpoint if args.strategy == "reinforce" else None,
            save_path=args.save or (ROOT / "checkpoints" / "router_bandit"),
        )


if __name__ == "__main__":
    main()
