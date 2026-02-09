"""Load evaluation set from JSONL."""
from __future__ import annotations

import json
from pathlib import Path


def load_eval_set(path: Path | str) -> list[dict]:
    """
    Load eval set from JSONL. Each line: {"query": str, "query_id"?: str, "reference_answer"?: str, "gold_doc_ids"?: list[str]}.
    Returns list of dicts with keys query, query_id, reference_answer, gold_doc_ids (defaults: query_id=query, reference_answer="", gold_doc_ids=[]).
    """
    path = Path(path)
    out = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append({
                "query": row.get("query", row.get("q", "")),
                "query_id": row.get("query_id", row.get("id", str(i))),
                "reference_answer": row.get("reference_answer", row.get("ref", "")),
                "gold_doc_ids": set(row.get("gold_doc_ids", row.get("gold_doc_id", [])) or []),
            })
    return out
