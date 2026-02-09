"""Test eval set loader."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import load_eval_set


def test_load_eval_set(tmp_path: Path) -> None:
    p = tmp_path / "eval.jsonl"
    p.write_text(
        '{"query": "q1", "reference_answer": "a1"}\n'
        '{"query": "q2", "query_id": "x", "gold_doc_ids": ["d1"]}\n',
        encoding="utf-8",
    )
    out = load_eval_set(p)
    assert len(out) == 2
    assert out[0]["query"] == "q1"
    assert out[0]["reference_answer"] == "a1"
    assert out[0]["gold_doc_ids"] == set()
    assert out[1]["query_id"] == "x"
    assert out[1]["gold_doc_ids"] == {"d1"}


def test_load_eval_set_missing_returns_empty() -> None:
    out = load_eval_set(ROOT / "nonexistent_eval.jsonl")
    assert out == []
