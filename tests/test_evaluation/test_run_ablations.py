"""Test run_ablations (smoke: run baseline and rules, check table)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import run_ablations, format_table, compute_metrics, run_strategy, load_eval_set


def test_run_ablations_baseline_and_rules(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text('{"query": "test query", "query_id": "1"}\n', encoding="utf-8")
    rows = run_ablations(
        eval_set_path=eval_path,
        strategies=["baseline", "rules"],
        checkpoint_path=None,
        log_dir=None,
    )
    assert len(rows) == 2
    assert rows[0]["strategy"] == "baseline"
    assert rows[1]["strategy"] == "rules"
    assert not rows[0].get("skipped")
    assert not rows[1].get("skipped")
    assert "avg_chunks" in rows[0].get("metrics", {})
    table = format_table(rows)
    assert "baseline" in table
    assert "rules" in table


def test_compute_metrics() -> None:
    from src.schema import Chunk, RetrievalResult
    eval_set = [{"query": "q", "gold_doc_ids": {"d1"}, "reference_answer": "yes"}]
    results = [
        {
            "chunks": [Chunk(text="x", score=1.0, source="a", doc_id="d1")],
            "answer": "yes",
            "retrieval_results": [RetrievalResult(query="q", retriever_name="a", chunks=[], latency_seconds=0.1, cost=0.0)],
        },
    ]
    m = compute_metrics(results, eval_set)
    assert m["hit_at_5"] == 1.0
    assert m["mrr"] == 1.0
    assert m["exact_match"] == 1.0
    assert m["avg_chunks"] == 1.0
