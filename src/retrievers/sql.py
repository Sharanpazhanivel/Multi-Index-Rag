"""SQL / DuckDB retriever: query table, return rows as chunks."""
import time
from pathlib import Path

from src.schema import Chunk, RetrievalResult
from .base import BaseRetriever
from .index_store import index_path


class SQLRetriever(BaseRetriever):
    name = "structured"

    def __init__(self, index_dir: Path | str | None = None) -> None:
        self._db_path = index_dir if index_dir is not None else str(index_path("structured", "data.duckdb"))

    def retrieve(self, query: str, top_k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        path = Path(self._db_path) if not isinstance(self._db_path, Path) else self._db_path
        if not path.exists():
            return RetrievalResult(query=query, retriever_name=self.name, chunks=[], latency_seconds=time.perf_counter() - t0, cost=0.0)
        import duckdb
        con = duckdb.connect(str(path), read_only=True)
        chunks = []
        try:
            terms = [t.replace("'", "''") for t in query.replace(",", " ").replace(".", " ").split()[:5] if t]
            if terms:
                placeholders = " OR ".join(["text ILIKE ?" for _ in terms])
                params = [f"%{t}%" for t in terms] + [top_k]
                rows = con.execute(f"SELECT id, text FROM docs WHERE ({placeholders}) LIMIT ?", params).fetchall()
            else:
                rows = con.execute("SELECT id, text FROM docs LIMIT ?", [top_k]).fetchall()
            chunks = [Chunk(text=r[1], score=1.0, source=self.name, doc_id=str(r[0])) for r in rows]
        except Exception:
            try:
                rows = con.execute("SELECT id, text FROM docs LIMIT ?", [top_k]).fetchall()
                chunks = [Chunk(text=r[1], score=1.0, source=self.name, doc_id=str(r[0])) for r in rows]
            except Exception:
                pass
        con.close()
        return RetrievalResult(
            query=query,
            retriever_name=self.name,
            chunks=chunks,
            latency_seconds=time.perf_counter() - t0,
            cost=0.0,
        )
