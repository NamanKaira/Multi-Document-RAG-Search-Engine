"""
Text chunking with metadata preservation.
Uses LangChain's recursive character text splitter.
"""

from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.document import Document, DocumentChunk, SourceType


def create_chunker(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: List[str] = None
) -> RecursiveCharacterTextSplitter:
    """
    Create a recursive character text splitter.
    
    Args:
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        separators: Priority list of separators to use for splitting
    
    Returns:
        Configured text splitter
    """
    if separators is None:
        # Default separators in order of priority
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )


def chunk_document(
    document: Document,
    chunker: RecursiveCharacterTextSplitter = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[DocumentChunk]:
    """
    Split a document into chunks with metadata preservation.
    
    Args:
        document: Source document
        chunker: Pre-configured text splitter (optional)
        chunk_size: Chunk size if creating new chunker
        chunk_overlap: Overlap if creating new chunker
    
    Returns:
        List of DocumentChunk objects
    """
    if chunker is None:
        chunker = create_chunker(chunk_size, chunk_overlap)
    
    # Split the document content
    texts = chunker.split_text(document.content)
    
    chunks = []
    for i, text in enumerate(texts):
        chunk = DocumentChunk(
            parent_doc_id=document.source_id,
            chunk_index=i,
            content=text,
            metadata={
                "source_type": document.source_type,
                "document_title": document.title,
                "total_chunks": len(texts),
                **document.metadata  # Include parent document metadata
            }
        )
        chunks.append(chunk)
    
    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[DocumentChunk]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of documents to chunk
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
    
    Returns:
        List of all chunks from all documents
    """
    chunker = create_chunker(chunk_size, chunk_overlap)
    
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunker=chunker)
        all_chunks.extend(chunks)
    
    return all_chunks


def get_chunk_statistics(chunks: List[DocumentChunk]) -> dict:
    """
    Get statistics about chunks.
    
    Args:
        chunks: List of DocumentChunk objects
    
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "by_source_type": {}
        }
    
    sizes = [len(chunk.content) for chunk in chunks]
    
    # Count by source type
    by_type = {}
    for chunk in chunks:
        source_type = chunk.source_type
        if source_type:
            type_key = source_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
    
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(sizes) / len(sizes),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "by_source_type": by_type
    }
