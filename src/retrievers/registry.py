"""Registry of retrievers by name."""
from .base import BaseRetriever
from .bm25 import BM25Retriever
from .code import CodeRetriever
from .sql import SQLRetriever
from .vector import VectorRetriever

_REGISTRY: dict[str, BaseRetriever] = {}


def _init_registry() -> None:
    for r in [BM25Retriever(), VectorRetriever(), SQLRetriever(), CodeRetriever()]:
        _REGISTRY[r.name] = r


def register_retriever(name: str, retriever: BaseRetriever) -> None:
    _REGISTRY[name] = retriever


def get_retriever(name: str) -> BaseRetriever:
    if not _REGISTRY:
        _init_registry()
    return _REGISTRY[name]
