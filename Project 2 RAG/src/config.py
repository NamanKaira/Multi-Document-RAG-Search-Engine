"""
Centralized configuration for the RAG system.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


# API Keys
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")

# Model Configuration
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
GOOGLE_LLM_MODEL: str = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash")

# Provider selection
USE_GOOGLE_LLM: bool = os.getenv("USE_GOOGLE_LLM", "false").lower() == "true"

# Tavily Configuration
TAVILY_MAX_RESULTS: int = int(os.getenv("TAVILY_MAX_RESULTS", "5"))

# Chunking Configuration
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Retrieval Configuration
DEFAULT_TOP_K: int = 5
MAX_CONTEXT_TOKENS: int = 4000

# FAISS Configuration
FAISS_INDEX_NAME: str = "document_index"


def validate_config() -> list:
    """
    Validate configuration and return list of missing/invalid items.
    
    Returns:
        List of error messages
    """
    errors = []
    
    # Check for at least one LLM provider
    if not OPENAI_API_KEY and not GOOGLE_API_KEY:
        errors.append("No API key set. Please set OPENAI_API_KEY or GOOGLE_API_KEY in .env file.")
    
    # OpenAI key required for embeddings (unless using alternative)
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is recommended for embeddings (best quality).")
    
    if not TAVILY_API_KEY:
        errors.append("TAVILY_API_KEY is not set. Please set it in .env file.")
    
    return errors


def get_faiss_index_path() -> Path:
    """Get the path to the FAISS index directory."""
    return FAISS_INDEX_DIR / FAISS_INDEX_NAME
