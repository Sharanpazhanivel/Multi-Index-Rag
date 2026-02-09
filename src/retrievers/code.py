"""Code index retriever: same as vector but loads from code index path."""
import time
from pathlib import Path

import numpy as np

from src.schema import Chunk, RetrievalResult
from .base import BaseRetriever
from .index_store import index_path


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CodeRetriever(BaseRetriever):
    name = "code"

    def __init__(self, index_dir: Path | str | None = None) -> None:
        self._embeddings = None
        self._doc_ids = None
        self._doc_texts = None
        self._embed_fn = None
        self._index_dir = Path(index_dir) if index_dir else None

    def _load(self) -> None:
        if self._embeddings is not None:
            return
        base = self._index_dir or index_path("code", "vectors.npz").parent
        npz_path = base / "code_vectors.npz"
        meta_path = base / "code_docs.json"
        if not npz_path.exists():
            return
        data = np.load(npz_path)
        self._embeddings = data["embeddings"]
        import json
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._doc_ids = meta.get("doc_ids", [])
            self._doc_texts = meta.get("doc_texts", [])
        else:
            self._doc_ids = [str(i) for i in range(len(self._embeddings))]
            self._doc_texts = [""] * len(self._embeddings)
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_fn = SentenceTransformer("all-MiniLM-L6-v2").encode
        except ImportError:
            self._embed_fn = None

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        self._load()
        if self._embeddings is None or self._embed_fn is None:
            return RetrievalResult(query=query, retriever_name=self.name, chunks=[], latency_seconds=time.perf_counter() - t0, cost=0.0)
        q_vec = self._embed_fn([query], convert_to_numpy=True).flatten()
        scores = np.array([_cosine_sim(q_vec, self._embeddings[i]) for i in range(len(self._embeddings))])
        top_indices = np.argsort(-scores)[:top_k]
        chunks = []
        for i in top_indices:
            i = int(i)
            doc_id = self._doc_ids[i] if i < len(self._doc_ids) else str(i)
            text = self._doc_texts[i] if i < len(self._doc_texts) else ""
            score = float(scores[i])
            chunks.append(Chunk(text=text, score=score, source=self.name, doc_id=doc_id))
        return RetrievalResult(
            query=query,
            retriever_name=self.name,
            chunks=chunks,
            latency_seconds=time.perf_counter() - t0,
            cost=0.0,
        )
