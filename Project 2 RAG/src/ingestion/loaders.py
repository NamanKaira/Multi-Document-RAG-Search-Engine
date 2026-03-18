"""
Multi-source document loaders using LangChain.
Supports PDF, Wikipedia, and text files.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WikipediaLoader,
)
from langchain_core.documents import Document as LangChainDocument

from src.ingestion.cleaning import clean_text
from src.models.document import Document, SourceType


def convert_langchain_doc(
    lc_doc: LangChainDocument,
    source_type: SourceType,
    title: Optional[str] = None
) -> Document:
    """
    Convert LangChain Document to our unified Document model.
    
    Args:
        lc_doc: LangChain document
        source_type: Type of source
        title: Optional title override
    
    Returns:
        Unified Document object
    """
    content = lc_doc.page_content or ""
    metadata = dict(lc_doc.metadata) if lc_doc.metadata else {}
    
    # Clean the content
    content = clean_text(content)
    
    # Determine title
    if title:
        doc_title = title
    elif "title" in metadata:
        doc_title = metadata["title"]
    elif "source" in metadata:
        doc_title = Path(metadata["source"]).stem
    else:
        doc_title = "Untitled"
    
    # Generate source_id from metadata if available
    source_id = metadata.get("source", str(hash(content[:100])))
    
    return Document(
        source_id=source_id,
        source_type=source_type,
        title=doc_title,
        content=content,
        metadata=metadata
    )


def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    """
    Load PDF file and convert to Document objects.
    
    Args:
        file_path: Path to PDF file
    
    Returns:
        List of Document objects (one per page)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    loader = PyPDFLoader(str(file_path))
    lc_docs = loader.load()
    
    title = file_path.stem
    documents = []
    
    for i, lc_doc in enumerate(lc_docs):
        # Add page number to metadata
        lc_doc.metadata["page_number"] = i + 1
        lc_doc.metadata["total_pages"] = len(lc_docs)
        
        doc = convert_langchain_doc(lc_doc, SourceType.PDF, title)
        doc.metadata["page_number"] = i + 1
        documents.append(doc)
    
    return documents


def load_text_file(file_path: Union[str, Path]) -> List[Document]:
    """
    Load text file and convert to Document object.
    
    Args:
        file_path: Path to text file
    
    Returns:
        List containing single Document object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    loader = TextLoader(str(file_path), encoding='utf-8')
    lc_docs = loader.load()
    
    title = file_path.stem
    documents = []
    
    for lc_doc in lc_docs:
        doc = convert_langchain_doc(lc_doc, SourceType.TEXT, title)
        documents.append(doc)
    
    return documents


def load_wikipedia(
    query: str,
    load_max_docs: int = 1,
    lang: str = "en"
) -> List[Document]:
    """
    Load Wikipedia page(s) by search query.
    
    Args:
        query: Search query or page title
        load_max_docs: Maximum number of documents to load
        lang: Wikipedia language code
    
    Returns:
        List of Document objects
    """
    loader = WikipediaLoader(
        query=query,
        load_max_docs=load_max_docs,
        lang=lang,
        load_all_available_meta=True
    )
    
    try:
        lc_docs = loader.load()
    except Exception as e:
        print(f"Error loading Wikipedia page for '{query}': {e}")
        return []
    
    documents = []
    for lc_doc in lc_docs:
        title = lc_doc.metadata.get("title", query)
        doc = convert_langchain_doc(lc_doc, SourceType.WIKIPEDIA, title)
        documents.append(doc)
    
    return documents


def load_directory(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None
) -> List[Document]:
    """
    Load all supported documents from a directory.
    
    Args:
        directory: Path to directory
        extensions: List of file extensions to include (default: .pdf, .txt, .md)
    
    Returns:
        List of Document objects
    """
    if extensions is None:
        extensions = [".pdf", ".txt", ".md"]
    
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    documents = []
    
    for ext in extensions:
        for file_path in directory.glob(f"**/*{ext}"):
            try:
                if ext == ".pdf":
                    docs = load_pdf(file_path)
                elif ext in [".txt", ".md"]:
                    docs = load_text_file(file_path)
                else:
                    continue
                
                documents.extend(docs)
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents


def load_documents(
    sources: List[Union[str, Path, dict]]
) -> List[Document]:
    """
    Load documents from multiple sources.
    
    Args:
        sources: List of sources. Each can be:
            - str/Path: File or directory path
            - dict: {"type": "wikipedia", "query": "..."}
    
    Returns:
        List of all loaded Document objects
    """
    documents = []
    
    for source in sources:
        if isinstance(source, dict):
            # Wikipedia query
            if source.get("type") == "wikipedia":
                query = source.get("query", "")
                if query:
                    docs = load_wikipedia(
                        query=query,
                        load_max_docs=source.get("load_max_docs", 1)
                    )
                    documents.extend(docs)
        else:
            # File or directory path
            path = Path(source)
            
            if path.is_dir():
                docs = load_directory(path)
                documents.extend(docs)
            elif path.is_file():
                if path.suffix == ".pdf":
                    docs = load_pdf(path)
                elif path.suffix in [".txt", ".md"]:
                    docs = load_text_file(path)
                else:
                    print(f"Unsupported file type: {path}")
                    continue
                documents.extend(docs)
    
    return documents


def get_document_stats(documents: List[Document]) -> dict:
    """
    Get statistics about loaded documents.
    
    Args:
        documents: List of Document objects
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_documents": len(documents),
        "by_type": {},
        "total_chars": 0,
        "total_words": 0,
    }
    
    for doc in documents:
        # Count by type
        doc_type = doc.source_type.value
        stats["by_type"][doc_type] = stats["by_type"].get(doc_type, 0) + 1
        
        # Content stats
        content = doc.content or ""
        stats["total_chars"] += len(content)
        stats["total_words"] += len(content.split())
    
    return stats
