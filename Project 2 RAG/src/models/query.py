"""
Query classification and routing models.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.models.document import QueryType


@dataclass
class QueryClassification:
    """
    Result of query classification.
    
    Attributes:
        query_type: Classified type (document/web/hybrid)
        confidence: Confidence score (0-1)
        reasoning: Explanation for classification
        suggested_sources: Which sources to query
    """
    query_type: QueryType
    confidence: float = 1.0
    reasoning: str = ""
    suggested_sources: List[str] = None
    
    def __post_init__(self):
        if self.suggested_sources is None:
            if self.query_type == QueryType.DOCUMENT:
                self.suggested_sources = ["faiss"]
            elif self.query_type == QueryType.WEB:
                self.suggested_sources = ["tavily"]
            else:  # hybrid
                self.suggested_sources = ["faiss", "tavily"]


@dataclass
class SearchRequest:
    """
    Internal representation of a search request.
    
    Attributes:
        query: User query text
        top_k: Number of results to retrieve per source
        use_documents: Whether to search FAISS index
        use_web: Whether to search web
        filters: Optional metadata filters
    """
    query: str
    top_k: int = 5
    use_documents: bool = True
    use_web: bool = False
    filters: Optional[dict] = None
    
    @classmethod
    def from_classification(
        cls,
        query: str,
        classification: QueryClassification,
        top_k: int = 5
    ) -> "SearchRequest":
        """Create search request from classification result."""
        return cls(
            query=query,
            top_k=top_k,
            use_documents="faiss" in classification.suggested_sources,
            use_web="tavily" in classification.suggested_sources
        )
