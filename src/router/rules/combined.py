"""Combine keyword + centroid (keyword first, centroid fallback for default)."""
from src.schema import RouterDecision
from src.router.base import BaseRouter
from .keyword_router import KeywordRouter
from .centroid_router import CentroidRouter


class CombinedRulesRouter(BaseRouter):
    """Keyword rules first; when keyword returns default (no SQL/code match), use centroid."""

    def __init__(
        self,
        use_centroid_fallback: bool = True,
        centroid_path: str | None = None,
        embed_fn=None,
    ) -> None:
        self.keyword = KeywordRouter()
        self.centroid = CentroidRouter(centroid_path=centroid_path, embed_fn=embed_fn)
        self.use_centroid_fallback = use_centroid_fallback

    def route(self, query: str, metadata: dict | None = None) -> RouterDecision:
        decision = self.keyword.route(query, metadata)
        # Use centroid only when keyword fell back to default (no SQL/code match)
        if self.use_centroid_fallback and decision.metadata and decision.metadata.get("rule") == "default":
            return self.centroid.route(query, metadata)
        return decision
