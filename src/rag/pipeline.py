"""RAG pipeline: optional router -> log -> retrieve -> re-rank -> LLM."""
from pathlib import Path
from typing import Callable

from src.retrievers import get_retriever
from src.schema import Chunk, RouterDecision, RetrievalResult
from src.rag.reranker import merge_and_rerank
from src.rag.baseline import baseline_retrieve
from src.rag.llm import build_context, default_llm_client
from src.logging.router_logger import log_router_decision, now_seconds

def _default_log_dir() -> Path | None:
    try:
        from config.settings import LOG_DIR
        return LOG_DIR
    except Exception:
        return None


class RAGPipeline:
    """Run retrieval from router decision (or baseline), with logging, re-rank, and optional LLM."""

    def __init__(
        self,
        router=None,
        log_dir: Path | str | None = None,
        top_k_per: int = 10,
        merge_top_k: int = 10,
        llm_client: Callable[[str, str], str] | None = None,
    ) -> None:
        self.router = router
        self.log_dir = Path(log_dir) if log_dir else _default_log_dir()
        self.top_k_per = top_k_per
        self.merge_top_k = merge_top_k
        self.llm_client = llm_client if llm_client is not None else default_llm_client()

    def run(
        self,
        query: str,
        query_id: str | None = None,
    ) -> dict:
        """Route (if router set), log decision and latency, retrieve, re-rank, then generate answer if LLM set."""
        decision: RouterDecision | None = None
        t0 = now_seconds()
        if self.router:
            decision = self.router.route(query)
            latency = now_seconds() - t0
            if self.log_dir:
                log_router_decision(query, decision, latency, self.log_dir, query_id)
            retriever_names = decision.retriever_names
        else:
            from src.retrievers.registry import _REGISTRY
            if not _REGISTRY:
                from src.retrievers.registry import _init_registry
                _init_registry()
            retriever_names = list(_REGISTRY.keys())
            decision = RouterDecision(action_id=-1, retriever_names=retriever_names, metadata={"source": "baseline"})

        results = baseline_retrieve(query, retriever_names, top_k_per=self.top_k_per)
        if len(results) > 1:
            chunks = merge_and_rerank(results, top_k=self.merge_top_k)
        else:
            chunks = results[0].chunks if results else []

        context = build_context(chunks)
        answer = self.llm_client(query, context) if self.llm_client else ""
        return {
            "query": query,
            "query_id": query_id,
            "chunks": chunks,
            "answer": answer,
            "decision": decision,
            "retrieval_results": results,
        }

    def record_feedback(self, query: str, decision: RouterDecision, reward: float, query_id: str | None = None) -> None:
        """Record thumbs up/down or proxy reward for this query and decision."""
        from src.logging.feedback import record_feedback as _record
        if self.log_dir:
            _record(query, decision, reward, log_dir=self.log_dir, query_id=query_id)
