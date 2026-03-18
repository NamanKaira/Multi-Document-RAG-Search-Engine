"""
Context assembly for RAG - combines document and web results.
"""

from typing import List, Tuple

import tiktoken

from src.config import MAX_CONTEXT_TOKENS
from src.models.document import DocumentChunk, RetrievedContext, WebSearchResult


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name for encoding
    
    Returns:
        Token count
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def format_document_chunks(chunks: List[DocumentChunk]) -> str:
    """
    Format document chunks for context.
    
    Args:
        chunks: List of DocumentChunk objects
    
    Returns:
        Formatted context string
    """
    if not chunks:
        return ""
    
    formatted = ["=== DOCUMENT CHUNKS ==="]
    
    for i, chunk in enumerate(chunks, 1):
        title = chunk.document_title
        chunk_num = chunk.chunk_index + 1
        total = chunk.metadata.get("total_chunks", "?")
        
        formatted.append(f"\n--- Chunk {i} ---")
        formatted.append(f"Source: {title} (Chunk {chunk_num}/{total})")
        formatted.append(f"Content: {chunk.content}")
    
    return "\n".join(formatted)


def format_web_results(results: List[WebSearchResult]) -> str:
    """
    Format web search results for context.
    
    Args:
        results: List of WebSearchResult objects
    
    Returns:
        Formatted context string
    """
    if not results:
        return ""
    
    formatted = ["=== WEB SEARCH RESULTS ==="]
    
    for i, result in enumerate(results, 1):
        formatted.append(f"\n--- Web Result {i} ---")
        formatted.append(f"Title: {result.title}")
        formatted.append(f"URL: {result.url}")
        formatted.append(f"Content: {result.content or result.snippet}")
    
    return "\n".join(formatted)


def assemble_context(
    retrieved: RetrievedContext,
    max_tokens: int = None,
    prioritize_docs: bool = True
) -> str:
    """
    Assemble context from retrieved documents and web results.
    
    Args:
        retrieved: RetrievedContext with chunks and web results
        max_tokens: Maximum tokens for context
        prioritize_docs: Whether to prioritize document chunks
    
    Returns:
        Formatted context string within token limit
    """
    max_tokens = max_tokens or MAX_CONTEXT_TOKENS
    
    # Format both sources
    doc_context = format_document_chunks(retrieved.document_chunks)
    web_context = format_web_results(retrieved.web_results)
    
    # Combine based on priority
    if prioritize_docs:
        full_context = f"{doc_context}\n\n{web_context}".strip()
    else:
        full_context = f"{web_context}\n\n{doc_context}".strip()
    
    # Truncate if necessary
    if count_tokens(full_context) > max_tokens:
        full_context = truncate_context(
            retrieved,
            max_tokens,
            prioritize_docs
        )
    
    return full_context


def truncate_context(
    retrieved: RetrievedContext,
    max_tokens: int,
    prioritize_docs: bool = True
) -> str:
    """
    Truncate context to fit within token limit.
    
    Args:
        retrieved: RetrievedContext
        max_tokens: Maximum tokens
        prioritize_docs: Whether to prioritize documents
    
    Returns:
        Truncated context string
    """
    context_parts = []
    current_tokens = 0
    
    # Reserve tokens for header
    header_tokens = 100
    available_tokens = max_tokens - header_tokens
    
    if prioritize_docs and retrieved.document_chunks:
        # Add document chunks first
        doc_header = "=== DOCUMENT CHUNKS ==="
        context_parts.append(doc_header)
        current_tokens += count_tokens(doc_header)
        
        for chunk in retrieved.document_chunks:
            chunk_text = f"\n--- Chunk ---\nSource: {chunk.document_title}\nContent: {chunk.content}"
            chunk_tokens = count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens < available_tokens * 0.7:  # 70% for docs
                context_parts.append(chunk_text)
                current_tokens += chunk_tokens
            else:
                break
        
        # Add web results with remaining budget
        if retrieved.web_results:
            web_header = "\n\n=== WEB SEARCH RESULTS ==="
            web_tokens = count_tokens(web_header)
            
            if current_tokens + web_tokens < available_tokens:
                context_parts.append(web_header)
                current_tokens += web_tokens
                
                for result in retrieved.web_results:
                    result_text = f"\n--- Web Result ---\nTitle: {result.title}\nContent: {result.snippet or result.content[:500]}"
                    result_tokens = count_tokens(result_text)
                    
                    if current_tokens + result_tokens < available_tokens:
                        context_parts.append(result_text)
                        current_tokens += result_tokens
                    else:
                        break
    
    else:
        # Similar logic but prioritize web results
        if retrieved.web_results:
            web_header = "=== WEB SEARCH RESULTS ==="
            context_parts.append(web_header)
            current_tokens += count_tokens(web_header)
            
            for result in retrieved.web_results[:3]:  # Limit web results
                result_text = f"\n--- Web Result ---\nTitle: {result.title}\nContent: {result.snippet or result.content[:500]}"
                result_tokens = count_tokens(result_text)
                
                if current_tokens + result_tokens < available_tokens * 0.5:
                    context_parts.append(result_text)
                    current_tokens += result_tokens
    
    return "\n".join(context_parts)


def get_source_summary(retrieved: RetrievedContext) -> str:
    """
    Get a summary of sources for display.
    
    Args:
        retrieved: RetrievedContext
    
    Returns:
        Human-readable source summary
    """
    parts = []
    
    if retrieved.document_chunks:
        # Group by document title
        doc_titles = set()
        for chunk in retrieved.document_chunks:
            doc_titles.add(chunk.document_title)
        
        parts.append(f"Documents ({len(doc_titles)} sources, {len(retrieved.document_chunks)} chunks)")
    
    if retrieved.web_results:
        parts.append(f"Web ({len(retrieved.web_results)} results)")
    
    if not parts:
        return "No sources retrieved"
    
    return " | ".join(parts)
