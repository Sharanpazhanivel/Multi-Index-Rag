"""Common schema for retrieved chunks."""
from pydantic import BaseModel


class Chunk(BaseModel):
    """Standardized chunk from any retriever."""
    text: str
    score: float
    source: str  # retriever/index name
    doc_id: str | None = None
    metadata: dict | None = None
