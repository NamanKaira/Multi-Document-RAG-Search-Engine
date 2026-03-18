"""
Document summarization for top-N document summaries.
Supports OpenAI and Google Gemini.
"""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    OPENAI_API_KEY, OPENAI_LLM_MODEL,
    GOOGLE_API_KEY, GOOGLE_LLM_MODEL, USE_GOOGLE_LLM
)
from src.models.document import DocumentChunk


def get_summary_llm(model: Optional[str] = None, temperature: float = 0.3):
    """Get LLM for summarization."""
    if USE_GOOGLE_LLM and GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or GOOGLE_LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature
        )
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or OPENAI_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature
        )
    return None


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that summarizes document content concisely."),
    ("human", """Please provide a brief summary (2-3 sentences) of the following document content:

Document: {document_title}
Content:
{content}

Summary:""")
])


def summarize_chunks(
    chunks: List[DocumentChunk],
    model: Optional[str] = None
) -> str:
    """
    Generate a summary from document chunks.
    
    Args:
        chunks: List of chunks from the same document
        model: LLM model name
    
    Returns:
        Summary text
    """
    if not chunks:
        return "No content to summarize."
    
    llm = get_summary_llm(model)
    if llm is None:
        return "Summary unavailable (API key not configured)"
    
    # Combine chunks (limit to first few to avoid token limits)
    combined_content = "\n\n".join([c.content for c in chunks[:5]])
    document_title = chunks[0].document_title
    
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    
    try:
        summary = chain.invoke({
            "document_title": document_title,
            "content": combined_content[:4000]  # Limit content length
        })
        return summary
    except Exception as e:
        return f"Error generating summary: {e}"


def get_top_document_summaries(
    chunks: List[DocumentChunk],
    top_n: int = 3
) -> List[dict]:
    """
    Get summaries for top N documents based on chunk count.
    
    Args:
        chunks: All retrieved chunks
        top_n: Number of top documents to summarize
    
    Returns:
        List of dicts with document info and summary
    """
    # Group chunks by document
    doc_chunks = {}
    for chunk in chunks:
        doc_id = chunk.parent_doc_id
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = {
                "title": chunk.document_title,
                "chunks": []
            }
        doc_chunks[doc_id]["chunks"].append(chunk)
    
    # Sort by number of chunks (relevance proxy) and take top N
    sorted_docs = sorted(
        doc_chunks.items(),
        key=lambda x: len(x[1]["chunks"]),
        reverse=True
    )[:top_n]
    
    summaries = []
    for doc_id, doc_data in sorted_docs:
        summary = summarize_chunks(doc_data["chunks"])
        summaries.append({
            "document_id": doc_id,
            "title": doc_data["title"],
            "chunk_count": len(doc_data["chunks"]),
            "summary": summary
        })
    
    return summaries


def format_summaries(summaries: List[dict]) -> str:
    """
    Format document summaries for display.
    
    Args:
        summaries: List of summary dicts
    
    Returns:
        Formatted string
    """
    if not summaries:
        return "No document summaries available."
    
    output = ["## Top Document Summaries\n"]
    
    for i, summary in enumerate(summaries, 1):
        output.append(f"**{i}. {summary['title']}** ({summary['chunk_count']} chunks)")
        output.append(f"{summary['summary']}\n")
    
    return "\n".join(output)
