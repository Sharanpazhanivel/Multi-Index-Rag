"""Router output: which retrievers to use."""
from pydantic import BaseModel


class RouterDecision(BaseModel):
    """Decision produced by the router."""
    action_id: int
    retriever_names: list[str]
    metadata: dict | None = None
