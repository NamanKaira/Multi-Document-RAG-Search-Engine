"""
Embedding configuration and factory.
"""

from langchain_openai import OpenAIEmbeddings

from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    """
    Get configured OpenAI embeddings instance.
    
    Returns:
        OpenAIEmbeddings instance
    
    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please set it in your .env file."
        )
    
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )


def get_embedding_dimension() -> int:
    """
    Get the dimension of the embedding model.
    
    Returns:
        Embedding dimension
    """
    # text-embedding-3-small = 1536 dimensions
    # text-embedding-3-large = 3072 dimensions
    # text-embedding-ada-002 = 1536 dimensions
    
    model_dims = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    return model_dims.get(OPENAI_EMBEDDING_MODEL, 1536)
