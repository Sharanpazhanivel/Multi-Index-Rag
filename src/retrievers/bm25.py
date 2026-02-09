"""BM25 retriever using rank_bm25 (local index)."""
import time
from pathlib import Path

from src.schema import Chunk, RetrievalResult
from .base import BaseRetriever
from .index_store import index_path


def _tokenize(text: str) -> list[str]:
    return text.lower().replace(".", " ").replace(",", " ").split()


class BM25Retriever(BaseRetriever):
    name = "general"

    def __init__(self, index_dir: Path | str | None = None) -> None:
        self._index = None
        self._doc_ids = None
        self._doc_texts = None
        self._index_dir = Path(index_dir) if index_dir else None

    def _load(self) -> None:
        if self._index is not None:
            return
        path = self._index_dir or index_path("general", "bm25.pkl")
        if not path.exists():
            return
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._index = data["bm25"]
        self._doc_ids = data.get("doc_ids", [])
        self._doc_texts = data.get("doc_texts", [])

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        self._load()
        if self._index is None:
            return RetrievalResult(query=query, retriever_name=self.name, chunks=[], latency_seconds=time.perf_counter() - t0, cost=0.0)
        from rank_bm25 import BM25Okapi
        query_tokens = _tokenize(query)
        scores = self._index.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        chunks = []
        for i in top_indices:
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
