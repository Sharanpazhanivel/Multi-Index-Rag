"""Phase 1: Run rules-based router, log decisions + feedback for Phase 2 labels.
Usage:
  Put queries in data/labels/queries.jsonl (one JSON object per line: {"query": "..."} or {"query": "...", "id": "..."})
  Or use data/labels/queries.txt (one query per line).
  Run: python scripts/phase1_collect_labels.py
  Logs: data/processed/logs/router_log.jsonl
  Optional offline eval (try all actions, score, output best action):
    python scripts/phase1_collect_labels.py --offline-eval
  Produces: data/labels/routing_labels.jsonl with query_id, query, best_action_id (for Phase 2 training).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_LABELS, DATA_PROCESSED, LOG_DIR
from src.router.rules import CombinedRulesRouter
from src.rag import RAGPipeline
from src.router.bandit.arms import get_action_retrievers


def load_queries(queries_path: Path) -> list[dict]:
    """Load list of {query, id?} from .jsonl or .txt."""
    if not queries_path.exists():
        return []
    out = []
    if queries_path.suffix == ".jsonl":
        with open(queries_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out.append({"query": obj.get("query", obj.get("q", "")), "id": obj.get("id", obj.get("query_id"))})
    else:
        with open(queries_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                q = line.strip()
                if q:
                    out.append({"query": q, "id": str(i)})
    return out


def run_router_and_log(queries: list[dict], log_dir: Path, use_centroid: bool = True) -> None:
    """Run CombinedRulesRouter + RAGPipeline for each query; log decisions."""
    centroid_path = DATA_PROCESSED / "centroids.json"
    embed_fn = None
    if centroid_path.exists():
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embed_fn = lambda q: model.encode(q, convert_to_numpy=True).tolist()
        except Exception:
            pass
    router = CombinedRulesRouter(use_centroid_fallback=use_centroid, centroid_path=str(centroid_path), embed_fn=embed_fn)
    pipeline = RAGPipeline(router=router, log_dir=log_dir)
    for item in queries:
        q = item["query"]
        qid = item.get("id")
        pipeline.run(query=q, query_id=qid)
    print(f"Logged {len(queries)} decisions to {log_dir / 'router_log.jsonl'}")


def offline_eval(queries: list[dict], labels_path: Path, log_dir: Path) -> None:
    """For each query try actions 0..4, run retrieval, score (e.g. by number of chunks), pick best. Write routing_labels.jsonl."""
    import yaml
    from src.rag import baseline_retrieve, merge_and_rerank

    actions_path = ROOT / "config" / "router_actions.yaml"
    with open(actions_path) as f:
        actions = yaml.safe_load(f)["actions"]
    action_ids = [a["id"] for a in actions]

    if labels_path.exists():
        labels_path.write_text("")

    for item in queries:
        q = item["query"]
        qid = item.get("id", "")
        best_action_id = 0
        best_score = -1.0
        for action_id in action_ids:
            retriever_names = get_action_retrievers(action_id)
            results = baseline_retrieve(q, retriever_names, top_k_per=10)
            chunks = merge_and_rerank(results, top_k=10) if len(results) > 1 else (results[0].chunks if results else [])
            score = len(chunks)
            if score > best_score:
                best_score = score
                best_action_id = action_id
        rec = {"query_id": qid, "query": q, "best_action_id": best_action_id}
        with open(labels_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(queries)} labels to {labels_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: rules router + logging; optional offline eval for labels.")
    ap.add_argument("--offline-eval", action="store_true", help="Try all actions per query, write best_action_id to routing_labels.jsonl")
    ap.add_argument("--queries", type=Path, default=DATA_LABELS / "queries.jsonl", help="Path to queries .jsonl or .txt")
    ap.add_argument("--no-centroid", action="store_true", help="Disable centroid fallback (keyword only)")
    args = ap.parse_args()

    log_dir = LOG_DIR or (DATA_PROCESSED / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(args.queries)
    if not queries:
        # Create sample queries file
        DATA_LABELS.mkdir(parents=True, exist_ok=True)
        sample = [{"query": "What is the refund policy?", "id": "1"}, {"query": "SELECT * FROM orders", "id": "2"}]
        with open(args.queries, "w") as f:
            for s in sample:
                f.write(json.dumps(s) + "\n")
        queries = load_queries(args.queries)
        print(f"Created sample {args.queries}. Re-run to log.")

    if args.offline_eval:
        labels_path = DATA_LABELS / "routing_labels.jsonl"
        if labels_path.exists():
            labels_path.write_text("")
        offline_eval(queries, labels_path, log_dir)
    else:
        run_router_and_log(queries, log_dir, use_centroid=not args.no_centroid)


if __name__ == "__main__":
    main()
