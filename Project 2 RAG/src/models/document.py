"""
Unified document schema for multi-source RAG system.
Supports local documents (PDF, text) and web search results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class SourceType(str, Enum):
    """Types of knowledge sources."""
    PDF = "pdf"
    TEXT = "text"
    WIKIPEDIA = "wikipedia"
    WEB = "web"


class QueryType(str, Enum):
    """Types of queries for routing."""
    DOCUMENT = "document"
    WEB = "web"
    HYBRID = "hybrid"


@dataclass
class Document:
    """
    Unified document model for all source types.
    
    Attributes:
        source_id: Unique identifier for the document
        source_type: Type of source (pdf, text, wikipedia, web)
        title: Document title
        content: Full document content
        metadata: Additional metadata (file path, URL, author, etc.)
    """
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_type: SourceType = SourceType.TEXT
    title: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)


@dataclass
class DocumentChunk:
    """
    A chunk of a document with metadata for retrieval.
    
    Attributes:
        chunk_id: Unique identifier for the chunk
        parent_doc_id: Reference to parent document
        chunk_index: Position of chunk within document
        content: Chunk text content
        metadata: Chunk-level metadata
    """
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_doc_id: str = ""
    chunk_index: int = 0
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def source_type(self) -> Optional[SourceType]:
        """Get source type from parent document metadata."""
        return self.metadata.get("source_type")
    
    @property
    def document_title(self) -> str:
        """Get parent document title."""
        return self.metadata.get("document_title", "Unknown")


@dataclass
class WebSearchResult:
    """
    Web search result from Tavily or similar service.
    
    Attributes:
        result_id: Unique identifier
        query: Original search query
        title: Page title
        url: Source URL
        snippet: Content snippet/summary
        content: Full content if available
        retrieved_at: Timestamp of retrieval
        score: Relevance score if available
    """
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    title: str = ""
    url: str = ""
    snippet: str = ""
    content: str = ""
    retrieved_at: datetime = field(default_factory=datetime.now)
    score: Optional[float] = None
    
    def to_document(self) -> Document:
        """Convert web result to Document for unified processing."""
        return Document(
            source_id=self.result_id,
            source_type=SourceType.WEB,
            title=self.title,
            content=self.content or self.snippet,
            metadata={
                "url": self.url,
                "query": self.query,
                "retrieved_at": self.retrieved_at.isoformat(),
                "score": self.score
            }
        )


@dataclass
class AnswerSource:
    """
    Source attribution for generated answers.
    
    Attributes:
        source_type: Type of source (document or web)
        source_id: Reference to source document/chunk/result
        citation_text: Formatted citation string
        relevance_score: Retrieval relevance score
        metadata: Additional source metadata
    """
    source_type: SourceType
    source_id: str
    citation_text: str
    relevance_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format_citation(self) -> str:
        """Format citation for display."""
        prefix = "[Doc]" if self.source_type in [SourceType.PDF, SourceType.TEXT, SourceType.WIKIPEDIA] else "[Web]"
        return f"{prefix} {self.citation_text}"


@dataclass
class RetrievedContext:
    """
    Container for retrieved context from all sources.
    
    Attributes:
        document_chunks: Chunks retrieved from FAISS
        web_results: Results from web search
        query_type: Classification of the original query
    """
    document_chunks: List[DocumentChunk] = field(default_factory=list)
    web_results: List[WebSearchResult] = field(default_factory=list)
    query_type: QueryType = QueryType.DOCUMENT
    
    def get_all_sources(self) -> List[AnswerSource]:
        """Convert all retrieved items to answer sources."""
        sources = []
        
        for chunk in self.document_chunks:
            sources.append(AnswerSource(
                source_type=chunk.source_type or SourceType.TEXT,
                source_id=chunk.chunk_id,
                citation_text=f"{chunk.document_title} – Chunk {chunk.chunk_index + 1}",
                metadata={"chunk_index": chunk.chunk_index}
            ))
        
        for result in self.web_results:
            sources.append(AnswerSource(
                source_type=SourceType.WEB,
                source_id=result.result_id,
                citation_text=f"Tavily: \"{result.title}\" ({result.url})",
                relevance_score=result.score,
                metadata={"url": result.url, "query": result.query}
            ))
        
        return sources
    
    def is_empty(self) -> bool:
        """Check if no context was retrieved."""
        return len(self.document_chunks) == 0 and len(self.web_results) == 0


@dataclass
class QueryResult:
    """
    Complete result of a RAG query.
    
    Attributes:
        query: Original user query
        answer: Generated answer text
        sources: List of sources used
        context: Retrieved context
        query_type: How the query was classified
        processing_time_ms: Time taken to process
    """
    query: str
    answer: str
    sources: List[AnswerSource] = field(default_factory=list)
    context: Optional[RetrievedContext] = None
    query_type: QueryType = QueryType.DOCUMENT
    processing_time_ms: Optional[float] = None
    
    def get_formatted_sources(self) -> str:
        """Get formatted source list for display."""
        if not self.sources:
            return "No sources available."
        return "\n".join([f"{i+1}. {s.format_citation()}" for i, s in enumerate(self.sources)])
