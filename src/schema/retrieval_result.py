"""Result of a single retriever call."""
from pydantic import BaseModel

from .chunk import Chunk


class RetrievalResult(BaseModel):
    """Query result from one retriever: chunks + observability."""
    query: str
    retriever_name: str
    chunks: list[Chunk]
    latency_seconds: float
    cost: float = 0.0
