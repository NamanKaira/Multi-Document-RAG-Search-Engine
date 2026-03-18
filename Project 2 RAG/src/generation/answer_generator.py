"""
Answer generation with citation support.
Supports both OpenAI and Google Gemini LLMs.
"""

import time
from typing import List, Optional, Union

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    OPENAI_API_KEY, OPENAI_LLM_MODEL,
    GOOGLE_API_KEY, GOOGLE_LLM_MODEL, USE_GOOGLE_LLM,
    MAX_CONTEXT_TOKENS
)
from src.models.document import (
    AnswerSource, DocumentChunk, QueryResult, 
    QueryType, RetrievedContext, WebSearchResult
)
from src.retrieval.context_assembly import assemble_context


def get_llm(model: Optional[str] = None, temperature: float = 0.3):
    """
    Get LLM instance based on configuration.
    
    Args:
        model: Model name override
        temperature: Generation temperature
    
    Returns:
        LLM instance (OpenAI or Google)
    """
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
    else:
        raise ValueError("No LLM API key configured. Set OPENAI_API_KEY or GOOGLE_API_KEY.")


# System prompt for citation-aware answer generation
CITATION_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on provided context.

Instructions:
1. Answer the question using ONLY the information in the provided context
2. If the context doesn't contain enough information, say so clearly
3. ALWAYS cite your sources using the format: [Doc] Document Name – Chunk N or [Web] Source Title
4. Place citations immediately after the relevant information
5. Be concise but complete
6. If both document and web sources are provided, synthesize information from both

Citation Format Examples:
- "Transformers use self-attention mechanisms [Doc] Attention Is All You Need – Chunk 2"
- "Recent advances include GPT-4 Turbo [Web] OpenAI Blog: GPT-4 Turbo Announcement"

Context:
{context}
"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CITATION_SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nPlease provide a cited answer:")
])


class AnswerGenerator:
    """
    Generates answers with source citations.
    Supports OpenAI and Google Gemini.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize the answer generator.
        
        Args:
            model: LLM model name
            temperature: Generation temperature
        """
        self.llm = get_llm(model, temperature)
        self.chain = ANSWER_PROMPT | self.llm | StrOutputParser()
    
    def generate(
        self,
        question: str,
        context: RetrievedContext,
        max_context_tokens: int = None
    ) -> QueryResult:
        """
        Generate an answer with citations.
        
        Args:
            question: User question
            context: Retrieved context (docs and/or web)
            max_context_tokens: Context token limit
        
        Returns:
            QueryResult with answer and sources
        """
        # Assemble context
        context_text = assemble_context(
            context,
            max_tokens=max_context_tokens or MAX_CONTEXT_TOKENS
        )
        
        if not context_text:
            return QueryResult(
                query=question,
                answer="I don't have enough information to answer this question. Please try uploading relevant documents or enabling web search.",
                sources=[],
                context=context,
                query_type=context.query_type
            )
        
        # Generate answer with retry for rate limits
        max_retries = 3
        retry_delay = 30  # seconds
        
        for attempt in range(max_retries):
            try:
                answer = self.chain.invoke({
                    "context": context_text,
                    "question": question
                })
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                raise
        
        # Get sources from context
        sources = context.get_all_sources()
        
        return QueryResult(
            query=question,
            answer=answer,
            sources=sources,
            context=context,
            query_type=context.query_type
        )


def generate_answer(
    question: str,
    document_chunks: List[DocumentChunk] = None,
    web_results: List[WebSearchResult] = None,
    query_type: QueryType = QueryType.DOCUMENT
) -> QueryResult:
    """
    Convenience function to generate an answer.
    
    Args:
        question: User question
        document_chunks: Retrieved document chunks
        web_results: Retrieved web results
        query_type: Type of query
    
    Returns:
        QueryResult with answer and sources
    """
    context = RetrievedContext(
        document_chunks=document_chunks or [],
        web_results=web_results or [],
        query_type=query_type
    )
    
    generator = AnswerGenerator()
    return generator.generate(question, context)


def format_answer_with_sources(result: QueryResult) -> str:
    """
    Format answer with sources for display.
    
    Args:
        result: QueryResult
    
    Returns:
        Formatted string
    """
    output = [result.answer]
    
    if result.sources:
        output.append("\n\n--- Sources ---")
        for i, source in enumerate(result.sources, 1):
            output.append(f"{i}. {source.format_citation()}")
    
    return "\n".join(output)
