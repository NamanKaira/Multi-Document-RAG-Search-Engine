"""
FAISS vector store management for document chunks.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument

from src.config import get_faiss_index_path
from src.models.document import DocumentChunk
from src.vectorstore.embeddings import get_embeddings


def chunks_to_langchain_docs(chunks: List[DocumentChunk]) -> List[LangChainDocument]:
    """
    Convert DocumentChunk objects to LangChain Document format.
    
    Args:
        chunks: List of DocumentChunk objects
    
    Returns:
        List of LangChain Document objects
    """
    return [
        LangChainDocument(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "parent_doc_id": chunk.parent_doc_id,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata
            }
        )
        for chunk in chunks
    ]


def langchain_doc_to_chunk(doc: LangChainDocument) -> DocumentChunk:
    """
    Convert LangChain Document back to DocumentChunk.
    
    Args:
        doc: LangChain Document with metadata
    
    Returns:
        DocumentChunk object
    """
    metadata = dict(doc.metadata)
    
    return DocumentChunk(
        chunk_id=metadata.pop("chunk_id", ""),
        parent_doc_id=metadata.pop("parent_doc_id", ""),
        chunk_index=metadata.pop("chunk_index", 0),
        content=doc.page_content,
        metadata=metadata
    )


class FAISSDocumentStore:
    """
    Manager for FAISS vector store operations.
    """
    
    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize the document store.
        
        Args:
            index_path: Path to store/load FAISS index
        """
        self.index_path = index_path or get_faiss_index_path()
        self.vectorstore: Optional[FAISS] = None
        self.embeddings = get_embeddings()
    
    def create_index(self, chunks: List[DocumentChunk]) -> "FAISSDocumentStore":
        """
        Create a new FAISS index from document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
        
        Returns:
            Self for method chaining
        """
        if not chunks:
            raise ValueError("Cannot create index from empty chunk list")
        
        # Convert to LangChain documents
        docs = chunks_to_langchain_docs(chunks)
        
        # Create FAISS index
        self.vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=self.embeddings
        )
        
        return self
    
    def add_documents(self, chunks: List[DocumentChunk]) -> "FAISSDocumentStore":
        """
        Add new documents to existing index.
        
        Args:
            chunks: List of DocumentChunk objects to add
        
        Returns:
            Self for method chaining
        """
        if not chunks:
            return self
        
        docs = chunks_to_langchain_docs(chunks)
        
        if self.vectorstore is None:
            return self.create_index(chunks)
        
        # Add to existing index
        self.vectorstore.add_documents(docs)
        
        return self
    
    def save(self, path: Optional[Path] = None) -> "FAISSDocumentStore":
        """
        Save the FAISS index to disk.
        
        Args:
            path: Optional custom save path
        
        Returns:
            Self for method chaining
        """
        if self.vectorstore is None:
            raise ValueError("No index to save. Create or load an index first.")
        
        save_path = path or self.index_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore.save_local(str(save_path))
        
        return self
    
    def load(self, path: Optional[Path] = None) -> "FAISSDocumentStore":
        """
        Load a FAISS index from disk.
        
        Args:
            path: Optional custom load path
        
        Returns:
            Self for method chaining
        
        Raises:
            FileNotFoundError: If index doesn't exist at path
        """
        load_path = path or self.index_path
        
        if not load_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {load_path}")
        
        self.vectorstore = FAISS.load_local(
            str(load_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        return self
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[dict] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search the index for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
        
        Returns:
            List of (DocumentChunk, score) tuples
        """
        if self.vectorstore is None:
            raise ValueError("No index loaded. Call load() or create_index() first.")
        
        # Perform similarity search
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )
        
        # Convert back to DocumentChunk objects
        results = []
        for doc, score in docs_with_scores:
            chunk = langchain_doc_to_chunk(doc)
            results.append((chunk, score))
        
        return results
    
    def is_index_loaded(self) -> bool:
        """Check if an index is currently loaded."""
        return self.vectorstore is not None
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore.index.ntotal


def index_documents(
    chunks: List[DocumentChunk],
    save: bool = True,
    index_path: Optional[Path] = None
) -> FAISSDocumentStore:
    """
    Convenience function to index documents and optionally save.
    
    Args:
        chunks: List of DocumentChunk objects
        save: Whether to save the index to disk
        index_path: Optional custom save path
    
    Returns:
        Configured FAISSDocumentStore
    """
    store = FAISSDocumentStore(index_path=index_path)
    store.create_index(chunks)
    
    if save:
        store.save()
    
    return store


def load_faiss_index(index_path: Optional[Path] = None) -> FAISSDocumentStore:
    """
    Convenience function to load a FAISS index.
    
    Args:
        index_path: Optional custom load path
    
    Returns:
        Configured FAISSDocumentStore with loaded index
    """
    store = FAISSDocumentStore(index_path=index_path)
    store.load()
    return store


def semantic_search(
    query: str,
    top_k: int = 5,
    index_path: Optional[Path] = None
) -> List[Tuple[DocumentChunk, float]]:
    """
    Convenience function to search the FAISS index.
    
    Args:
        query: Search query
        top_k: Number of results
        index_path: Optional custom index path
    
    Returns:
        List of (DocumentChunk, score) tuples
    """
    store = load_faiss_index(index_path)
    return store.search(query, top_k=top_k)
