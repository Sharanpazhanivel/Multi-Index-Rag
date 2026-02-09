"""Phase 0: Build indexes for general, technical, structured, code.
Usage:
  Put data in data/raw/: general.jsonl, technical.jsonl, structured.jsonl, code.jsonl
  (each line: {"id": "...", "text": "..."} or use .txt with one doc per line).
  Run: python scripts/phase0_build_indexes.py
  Output: data/processed/general_bm25.pkl, technical_vectors.npz, technical_docs.json,
          structured_data.duckdb, code_vectors.npz, code_docs.json
  If raw files missing, builds small sample indexes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_RAW, DATA_PROCESSED
from src.retrievers.build_indexes import build_all_indexes


def main() -> None:
    build_all_indexes(data_raw=DATA_RAW, data_processed=DATA_PROCESSED)
    print("Phase 0 done. Indexes in", DATA_PROCESSED)


if __name__ == "__main__":
    main()
