"""
Query classification and routing logic.
Determines whether to use document search, web search, or both.
Supports OpenAI and Google Gemini.
"""

import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    OPENAI_API_KEY, OPENAI_LLM_MODEL,
    GOOGLE_API_KEY, GOOGLE_LLM_MODEL, USE_GOOGLE_LLM
)
from src.models.document import QueryType
from src.models.query import QueryClassification


def get_classifier_llm():
    """Get LLM for query classification."""
    if USE_GOOGLE_LLM and GOOGLE_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=GOOGLE_LLM_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
    return None


# Few-shot examples for query classification
CLASSIFICATION_EXAMPLES = """
Examples:

Query: "Explain how transformers work in NLP"
Type: document
Reasoning: This is a foundational concept that is well-documented in textbooks and papers.

Query: "What are the latest developments in GPT models?"
Type: web
Reasoning: This asks for recent information that requires up-to-date knowledge.

Query: "How does RAG compare with current LLM tools?"
Type: hybrid
Reasoning: This requires both foundational knowledge (RAG) and current information (latest tools).

Query: "What is the attention mechanism?"
Type: document
Reasoning: This is established knowledge found in academic literature.

Query: "Latest stock prices for Apple"
Type: web
Reasoning: This requires real-time financial data.

Query: "Summarize the key points from my uploaded research paper"
Type: document
Reasoning: This specifically references uploaded documents.
"""

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query classifier for a hybrid RAG system.
Your task is to classify user queries into one of three types:

1. "document" - Query about established knowledge, concepts, or uploaded documents
2. "web" - Query requiring real-time, current, or recent information
3. "hybrid" - Query that benefits from both document knowledge AND current information

Respond with ONLY the classification type (document, web, or hybrid).

{examples}"""),
    ("human", "Query: {query}\nType:")
])


def classify_query_llm(query: str) -> QueryClassification:
    """
    Classify query using LLM.
    
    Args:
        query: User query
    
    Returns:
        QueryClassification with type and reasoning
    """
    llm = get_classifier_llm()
    
    if llm is None:
        # Fallback to heuristic if no API key
        return classify_query_heuristic(query)
    
    try:
        chain = CLASSIFICATION_PROMPT | llm | StrOutputParser()
        
        result = chain.invoke({
            "query": query,
            "examples": CLASSIFICATION_EXAMPLES
        })
        
        # Parse result
        result_lower = result.strip().lower()
        
        if "hybrid" in result_lower:
            query_type = QueryType.HYBRID
        elif "web" in result_lower:
            query_type = QueryType.WEB
        else:
            query_type = QueryType.DOCUMENT
        
        return QueryClassification(
            query_type=query_type,
            confidence=0.9,
            reasoning=f"LLM classification: {result.strip()}"
        )
    
    except Exception as e:
        # Fallback to heuristic on error
        return classify_query_heuristic(query)


def classify_query_heuristic(query: str) -> QueryClassification:
    """
    Classify query using keyword heuristics (fallback method).
    
    Args:
        query: User query
    
    Returns:
        QueryClassification
    """
    query_lower = query.lower()
    
    # Web-indicating keywords (real-time, current, recent, today, latest)
    web_keywords = [
        "latest", "recent", "current", "today", "now", "news",
        "update", "just happened", "this week", "this month",
        "stock price", "weather", "election results", "score",
        "live", "breaking", "trending"
    ]
    
    # Document-indicating keywords (foundational, conceptual)
    doc_keywords = [
        "explain", "what is", "how does", "describe", "define",
        "according to my document", "in the paper", "the text says",
        "uploaded file", "my pdf", "the book"
    ]
    
    web_score = sum(1 for kw in web_keywords if kw in query_lower)
    doc_score = sum(1 for kw in doc_keywords if kw in query_lower)
    
    # Determine type based on scores
    if web_score > 0 and doc_score > 0:
        query_type = QueryType.HYBRID
        reasoning = "Both real-time and document keywords detected"
    elif web_score > 0:
        query_type = QueryType.WEB
        reasoning = f"Real-time keywords detected: {web_score} matches"
    else:
        query_type = QueryType.DOCUMENT
        reasoning = "Defaulting to document search"
    
    return QueryClassification(
        query_type=query_type,
        confidence=0.7 if web_score > 0 or doc_score > 0 else 0.5,
        reasoning=reasoning
    )


def classify_query(
    query: str,
    use_llm: bool = True
) -> QueryClassification:
    """
    Classify a query to determine search strategy.
    
    Args:
        query: User query
        use_llm: Whether to use LLM classification (fallback to heuristic)
    
    Returns:
        QueryClassification with routing decision
    """
    has_llm = OPENAI_API_KEY or (USE_GOOGLE_LLM and GOOGLE_API_KEY)
    if use_llm and has_llm:
        return classify_query_llm(query)
    else:
        return classify_query_heuristic(query)


def should_search_web(classification: QueryClassification) -> bool:
    """Check if web search should be used."""
    return classification.query_type in [QueryType.WEB, QueryType.HYBRID]


def should_search_documents(classification: QueryClassification) -> bool:
    """Check if document search should be used."""
    return classification.query_type in [QueryType.DOCUMENT, QueryType.HYBRID]
