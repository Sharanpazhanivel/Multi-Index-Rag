"""Run comparison: no routing, rules, learned, bandit; compute metrics; output table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.rag import RAGPipeline
from src.evaluation.eval_set import load_eval_set
from src.evaluation.retrieval_metrics import hit_at_k, mrr
from src.evaluation.answer_metrics import exact_match, f1_score
from src.evaluation.system_metrics import avg_latency, total_cost


def _make_router(strategy: str, checkpoint_path: Path | None, num_actions: int = 5):
    """Build router for given strategy. Returns None if not available (e.g. missing checkpoint)."""
    if strategy == "baseline":
        return None
    if strategy == "rules":
        from src.router.rules import CombinedRulesRouter
        return CombinedRulesRouter(use_centroid_fallback=False)
    if strategy == "learned":
        if not checkpoint_path or not checkpoint_path.exists():
            return None
        from src.router.learned import load_router
        return load_router(checkpoint_path)
    if strategy == "bandit_linucb":
        from src.router.bandit import BanditRouter
        return BanditRouter(strategy="linucb", num_actions=num_actions, embed_fn=None)
    if strategy == "bandit_epsilon":
        from src.router.learned import LearnedRouter
        from src.router.bandit import BanditRouter
        base = LearnedRouter(num_actions=num_actions, model_path=checkpoint_path) if checkpoint_path and checkpoint_path.exists() else None
        return BanditRouter(strategy="epsilon_greedy", num_actions=num_actions, base_router=base, epsilon=0.1)
    if strategy == "bandit_reinforce":
        if not checkpoint_path or not checkpoint_path.exists():
            return None
        from src.router.bandit import BanditRouter
        return BanditRouter(strategy="reinforce", num_actions=num_actions, checkpoint_path=str(checkpoint_path))
    return None


def run_strategy(
    strategy: str,
    eval_set: list[dict],
    checkpoint_path: Path | None = None,
    num_actions: int = 5,
    log_dir: Path | None = None,
) -> list[dict]:
    """Run one strategy on full eval set. Returns list of pipeline outputs per query."""
    router = _make_router(strategy, checkpoint_path, num_actions)
    pipeline = RAGPipeline(router=router, log_dir=log_dir)
    results = []
    for item in eval_set:
        out = pipeline.run(query=item["query"], query_id=item.get("query_id"))
        out["_eval_item"] = item
        results.append(out)
    return results


def compute_metrics(results: list[dict], eval_set: list[dict]) -> dict[str, float]:
    """Aggregate metrics over results. Uses eval_set for gold_doc_ids and reference_answer."""
    hit5, mrr_list, em_list, f1_list, lat_list, cost_list = [], [], [], [], [], []
    for out, item in zip(results, eval_set):
        chunks = out.get("chunks", [])
        gold = item.get("gold_doc_ids") or set()
        ref = item.get("reference_answer", "")
        pred = out.get("answer", "")

        if gold:
            hit5.append(hit_at_k(chunks, gold, k=5))
            mrr_list.append(mrr(chunks, gold))
        if ref:
            em_list.append(exact_match(pred, ref))
            f1_list.append(f1_score(pred, ref))

        res = out.get("retrieval_results", [])
        if res:
            lat_list.append(avg_latency(res))
            cost_list.append(total_cost(res))

    metrics = {}
    n = len(results)
    if n:
        metrics["n_queries"] = float(n)
    if hit5:
        metrics["hit_at_5"] = sum(hit5) / len(hit5)
    if mrr_list:
        metrics["mrr"] = sum(mrr_list) / len(mrr_list)
    if em_list:
        metrics["exact_match"] = sum(em_list) / len(em_list)
    if f1_list:
        metrics["f1"] = sum(f1_list) / len(f1_list)
    if lat_list:
        metrics["avg_latency_sec"] = sum(lat_list) / len(lat_list)
    if cost_list:
        metrics["avg_cost"] = sum(cost_list) / len(cost_list)
    metrics["avg_chunks"] = (sum(len(out.get("chunks", [])) for out in results) / n) if n else 0.0
    return metrics


def run_ablations(
    eval_set_path: Path | str,
    strategies: list[str] | None = None,
    checkpoint_path: Path | str | None = None,
    num_actions: int = 5,
    log_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Run all strategies on eval set, compute metrics per strategy.
    Returns list of {strategy, metrics, skipped?}.
    """
    eval_set_path = Path(eval_set_path)
    eval_set = load_eval_set(eval_set_path)
    if not eval_set:
        return []

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    log_dir = Path(log_dir) if log_dir else None
    strategies = strategies or ["baseline", "rules", "learned", "bandit_linucb", "bandit_epsilon", "bandit_reinforce"]

    rows = []
    for strategy in strategies:
        router = _make_router(strategy, checkpoint_path, num_actions)
        if strategy not in ("baseline", "rules", "bandit_linucb") and router is None:
            rows.append({"strategy": strategy, "metrics": {}, "skipped": True})
            continue
        results = run_strategy(strategy, eval_set, checkpoint_path, num_actions, log_dir)
        metrics = compute_metrics(results, eval_set)
        rows.append({"strategy": strategy, "metrics": metrics, "skipped": False})
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    """Format ablation rows as a text table."""
    if not rows:
        return "No results."
    metric_keys = []
    for r in rows:
        if not r.get("skipped") and r.get("metrics"):
            for k in r["metrics"]:
                if k not in metric_keys:
                    metric_keys.append(k)
    keys = ["strategy"] + metric_keys
    col = 14
    header = " | ".join(k[:col].ljust(col) for k in keys)
    sep = "-+-".join("-" * col for _ in keys)
    lines = [header, sep]
    for r in rows:
        if r.get("skipped"):
            lines.append(r["strategy"][:col].ljust(col) + " | (skipped - missing checkpoint or config)")
            continue
        cells = [r["strategy"][:col].ljust(col)]
        for k in keys[1:]:
            v = r.get("metrics", {}).get(k)
            if v is None:
                cells.append("".ljust(col))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}".rjust(col))
            else:
                cells.append(f"{str(v)}".rjust(col)[:col].ljust(col))
        lines.append(" | ".join(cells))
    return "\n".join(lines)
