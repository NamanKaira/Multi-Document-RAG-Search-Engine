"""
Tavily web search integration.
"""

from typing import List, Optional

from langchain_community.tools import TavilySearchResults

from src.config import TAVILY_API_KEY, TAVILY_MAX_RESULTS
from src.models.document import WebSearchResult


class WebSearchClient:
    """
    Client for web search using Tavily.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = None,
        include_raw_content: bool = True
    ):
        """
        Initialize the web search client.
        
        Args:
            api_key: Tavily API key (defaults to config)
            max_results: Maximum results per query (defaults to config)
            include_raw_content: Whether to fetch full page content
        """
        self.api_key = api_key or TAVILY_API_KEY
        self.max_results = max_results or TAVILY_MAX_RESULTS
        self.include_raw_content = include_raw_content
        
        if not self.api_key:
            raise ValueError(
                "Tavily API key is required. Set TAVILY_API_KEY in .env file."
            )
        
        self.search_tool = TavilySearchResults(
            max_results=self.max_results,
            include_raw_content=include_raw_content,
            api_key=self.api_key
        )
    
    def search(self, query: str) -> List[WebSearchResult]:
        """
        Execute web search query.
        
        Args:
            query: Search query
        
        Returns:
            List of WebSearchResult objects
        """
        try:
            # Execute search
            results = self.search_tool.invoke({"query": query})
            
            # Convert to WebSearchResult objects
            web_results = []
            for result in results:
                web_result = WebSearchResult(
                    query=query,
                    title=result.get("title", "Untitled"),
                    url=result.get("url", ""),
                    snippet=result.get("content", ""),
                    content=result.get("raw_content", result.get("content", "")),
                    score=result.get("score")
                )
                web_results.append(web_result)
            
            return web_results
        
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if web search is properly configured."""
        return self.api_key is not None and len(self.api_key) > 0


def search_web(query: str, max_results: Optional[int] = None) -> List[WebSearchResult]:
    """
    Convenience function for web search.
    
    Args:
        query: Search query
        max_results: Override default max results
    
    Returns:
        List of WebSearchResult objects
    """
    try:
        client = WebSearchClient(max_results=max_results)
        return client.search(query)
    except ValueError as e:
        print(f"Web search not available: {e}")
        return []


def format_web_results_for_context(results: List[WebSearchResult]) -> str:
    """
    Format web search results for inclusion in LLM context.
    
    Args:
        results: List of WebSearchResult objects
    
    Returns:
        Formatted context string
    """
    if not results:
        return ""
    
    formatted = ["=== WEB SEARCH RESULTS ==="]
    
    for i, result in enumerate(results, 1):
        formatted.append(f"\n--- Result {i} ---")
        formatted.append(f"Title: {result.title}")
        formatted.append(f"URL: {result.url}")
        formatted.append(f"Content: {result.content or result.snippet}")
    
    return "\n".join(formatted)
