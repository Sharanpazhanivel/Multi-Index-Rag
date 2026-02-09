"""Build indexes from data/raw and save to data/processed."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from .index_store import _processed_dir, index_path


def _tokenize(text: str) -> list[str]:
    return text.lower().replace(".", " ").replace(",", " ").split()


def load_docs_jsonl(path: Path) -> list[tuple[str, str]]:
    """Load (doc_id, text) from JSONL. Each line: {"id": "...", "text": "..."} or {"doc_id": "...", "content": "..."}."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("id", obj.get("doc_id", str(i)))
            text = obj.get("text", obj.get("content", obj.get("body", "")))
            out.append((str(doc_id), text))
    return out


def load_docs_txt(path: Path) -> list[tuple[str, str]]:
    """Load (doc_id, text) from plain text: one doc per line or whole file as one doc."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return []
    return [(str(i), line.strip()) for i, line in enumerate(lines) if line.strip()]


def build_bm25_index(docs: list[tuple[str, str]], output_path: Path) -> None:
    """Build BM25 index and save to output_path (pickle)."""
    from rank_bm25 import BM25Okapi
    doc_ids = [d[0] for d in docs]
    doc_texts = [d[1] for d in docs]
    tokenized = [_tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "doc_ids": doc_ids, "doc_texts": doc_texts}, f)


def build_vector_index(docs: list[tuple[str, str]], output_npz: Path, output_meta: Path) -> None:
    """Embed docs, save embeddings and metadata."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers required for vector index. pip install sentence-transformers")
    doc_ids = [d[0] for d in docs]
    doc_texts = [d[1] for d in docs]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(doc_texts, convert_to_numpy=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.savez_compressed(output_npz, embeddings=embeddings)
    with open(output_meta, "w") as f:
        json.dump({"doc_ids": doc_ids, "doc_texts": doc_texts}, f, indent=0)


def build_duckdb_index(docs: list[tuple[str, str]], output_path: Path) -> None:
    """Create DuckDB table docs(id, text) and insert docs."""
    import duckdb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    con = duckdb.connect(str(output_path))
    con.execute("CREATE TABLE docs(id VARCHAR, text VARCHAR)")
    for doc_id, text in docs:
        con.execute("INSERT INTO docs VALUES (?, ?)", [doc_id, text])
    con.close()


def build_all_indexes(data_raw: Path | None = None, data_processed: Path | None = None) -> None:
    """Load from data_raw, build indexes, write to data_processed."""
    raw = data_raw or _processed_dir().parent / "raw"
    proc = data_processed or _processed_dir()
    proc.mkdir(parents=True, exist_ok=True)

    # General (BM25): general.jsonl or general.txt
    for name, ext in [("general", "jsonl"), ("general", "txt")]:
        path = raw / f"general.{ext}"
        if path.exists():
            docs = load_docs_jsonl(path) if ext == "jsonl" else load_docs_txt(path)
            if docs:
                build_bm25_index(docs, proc / "general_bm25.pkl")
                print(f"Built BM25 index: {len(docs)} docs -> general_bm25.pkl")
            break
    else:
        # Sample
        sample = [("0", "Refund policy: 30 days full refund."), ("1", "Contact support for help.")]
        build_bm25_index(sample, proc / "general_bm25.pkl")
        print("Built sample BM25 index (general_bm25.pkl)")

    # Technical (vector)
    for name, ext in [("technical", "jsonl"), ("technical", "txt")]:
        path = raw / f"technical.{ext}"
        if path.exists():
            docs = load_docs_jsonl(path) if ext == "jsonl" else load_docs_txt(path)
            if docs:
                build_vector_index(docs, proc / "technical_vectors.npz", proc / "technical_docs.json")
                print(f"Built vector index: {len(docs)} docs -> technical_vectors.npz")
            break
    else:
        sample = [("0", "API authentication uses OAuth2."), ("1", "Technical documentation here.")]
        build_vector_index(sample, proc / "technical_vectors.npz", proc / "technical_docs.json")
        print("Built sample vector index (technical_vectors.npz)")

    # Structured (DuckDB)
    for name, ext in [("structured", "jsonl"), ("structured", "csv")]:
        path = raw / f"structured.{ext}"
        if path.exists():
            if ext == "jsonl":
                docs = load_docs_jsonl(path)
            else:
                import csv
                docs = []
                with open(path) as f:
                    for i, row in enumerate(csv.DictReader(f)):
                        text = row.get("text", row.get("content", str(dict(row))))
                        docs.append((str(i), text))
            if docs:
                build_duckdb_index(docs, proc / "structured_data.duckdb")
                print(f"Built DuckDB index: {len(docs)} rows -> structured_data.duckdb")
            break
    else:
        sample = [
            ("0", "orders table: columns id (int), customer_id (int), total (decimal), status (varchar). Use SELECT * FROM orders WHERE status = 'pending' for pending orders."),
            ("1", "users table: columns id (int), name (varchar), email (varchar), active (boolean). Use SELECT id, name FROM users WHERE active = true for active users."),
            ("2", "products table: columns id (int), name (varchar), price (decimal). JOIN with orders to get order details. Example: SELECT p.name, o.total FROM orders o JOIN products p ON o.product_id = p.id."),
            ("3", "SQL tips: Use WHERE to filter rows, ORDER BY to sort, LIMIT to restrict results. COUNT(*) returns number of rows. GROUP BY for aggregations."),
        ]
        build_duckdb_index(sample, proc / "structured_data.duckdb")
        print("Built sample DuckDB (structured_data.duckdb)")

    # Code (vector, separate index)
    for name, ext in [("code", "jsonl"), ("code", "txt")]:
        path = raw / f"code.{ext}"
        if path.exists():
            docs = load_docs_jsonl(path) if ext == "jsonl" else load_docs_txt(path)
            if docs:
                build_vector_index(docs, proc / "code_vectors.npz", proc / "code_docs.json")
                print(f"Built code index: {len(docs)} docs -> code_vectors.npz")
            break
    else:
        sample = [("0", "def get_user(id): return db.query(id)"), ("1", "class ApiClient: pass")]
        build_vector_index(sample, proc / "code_vectors.npz", proc / "code_docs.json")
        print("Built sample code index (code_vectors.npz)")
