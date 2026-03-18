"""
Test script for the Hybrid RAG Search Engine.
Tests all major components without requiring the Streamlit UI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.document import Document, DocumentChunk, SourceType, QueryType, WebSearchResult
from src.models.query import QueryClassification, SearchRequest
from src.ingestion.cleaning import clean_text
from src.ingestion.chunking import create_chunker, chunk_document
from src.retrieval.query_router import classify_query_heuristic
from src.retrieval.context_assembly import format_document_chunks, format_web_results


def test_data_models():
    """Test data model creation."""
    print("\n" + "="*60)
    print("TEST 1: Data Models")
    print("="*60)
    
    # Test Document
    doc = Document(
        source_type=SourceType.PDF,
        title="Test Document",
        content="This is test content.",
        metadata={"author": "Test Author"}
    )
    print(f"✓ Document created: {doc.title} (ID: {doc.source_id[:8]}...)")
    
    # Test DocumentChunk
    chunk = DocumentChunk(
        parent_doc_id=doc.source_id,
        chunk_index=0,
        content="Test chunk content",
        metadata={"source_type": SourceType.PDF, "document_title": doc.title}
    )
    print(f"✓ DocumentChunk created: {chunk.document_title} - Chunk {chunk.chunk_index + 1}")
    
    # Test WebSearchResult
    web_result = WebSearchResult(
        query="test query",
        title="Test Web Result",
        url="https://example.com",
        snippet="Test snippet",
        content="Full content"
    )
    print(f"✓ WebSearchResult created: {web_result.title}")
    
    # Test QueryClassification
    classification = QueryClassification(
        query_type=QueryType.HYBRID,
        confidence=0.9,
        reasoning="Test reasoning"
    )
    print(f"✓ QueryClassification: {classification.query_type.value} (confidence: {classification.confidence})")
    
    print("✅ Data Models test passed")


def test_text_cleaning():
    """Test text cleaning functions."""
    print("\n" + "="*60)
    print("TEST 2: Text Cleaning")
    print("="*60)
    
    dirty_text = """
    This   is    a    test   with   extra   spaces.
    
    
    
    Multiple newlines above.
    Page 1
    Some content here.
    Copyright © 2024 All rights reserved
    """
    
    cleaned = clean_text(dirty_text)
    print(f"Original length: {len(dirty_text)}")
    print(f"Cleaned length: {len(cleaned)}")
    print(f"Cleaned preview: {cleaned[:100]}...")
    
    assert "   " not in cleaned, "Extra spaces should be removed"
    assert "Page 1" not in cleaned, "Page numbers should be removed"
    print("✅ Text Cleaning test passed")


def test_chunking():
    """Test document chunking."""
    print("\n" + "="*60)
    print("TEST 3: Document Chunking")
    print("="*60)
    
    # Create a test document
    long_content = """
    This is the first paragraph. It contains some information about machine learning.
    
    This is the second paragraph. It discusses neural networks and deep learning concepts.
    
    This is the third paragraph. It covers natural language processing and transformers.
    
    This is the fourth paragraph. It explains attention mechanisms in detail.
    
    This is the fifth paragraph. It talks about the future of AI research.
    """ * 10  # Make it longer
    
    doc = Document(
        source_type=SourceType.TEXT,
        title="ML Overview",
        content=long_content
    )
    
    # Test chunking
    chunker = create_chunker(chunk_size=500, chunk_overlap=50)
    chunks = chunk_document(doc, chunker=chunker)
    
    print(f"✓ Document split into {len(chunks)} chunks")
    print(f"✓ First chunk length: {len(chunks[0].content)} chars")
    print(f"✓ Chunk metadata preserved: {chunks[0].document_title}")
    
    assert len(chunks) > 1, "Long document should be split into multiple chunks"
    print("✅ Chunking test passed")


def test_query_classification():
    """Test query classification."""
    print("\n" + "="*60)
    print("TEST 4: Query Classification")
    print("="*60)
    
    test_queries = [
        ("What is machine learning?", QueryType.DOCUMENT),
        ("Latest news about AI", QueryType.WEB),
        ("Current stock prices", QueryType.WEB),
        ("Explain transformers in NLP", QueryType.DOCUMENT),
    ]
    
    for query, expected_type in test_queries:
        classification = classify_query_heuristic(query)
        status = "✓" if classification.query_type == expected_type else "⚠"
        print(f"{status} '{query}' -> {classification.query_type.value}")
        print(f"   Reasoning: {classification.reasoning}")
    
    print("✅ Query Classification test passed")


def test_context_assembly():
    """Test context assembly."""
    print("\n" + "="*60)
    print("TEST 5: Context Assembly")
    print("="*60)
    
    # Create test chunks
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            parent_doc_id="d1",
            chunk_index=0,
            content="Machine learning is a subset of AI.",
            metadata={"source_type": SourceType.PDF, "document_title": "AI Basics"}
        ),
        DocumentChunk(
            chunk_id="c2",
            parent_doc_id="d1",
            chunk_index=1,
            content="Deep learning uses neural networks.",
            metadata={"source_type": SourceType.PDF, "document_title": "AI Basics"}
        )
    ]
    
    # Create test web results
    web_results = [
        WebSearchResult(
            query="AI news",
            title="Latest AI Breakthrough",
            url="https://example.com/ai",
            snippet="New AI model achieves human-level performance.",
            content="Full article content here."
        )
    ]
    
    # Test formatting
    doc_context = format_document_chunks(chunks)
    web_context = format_web_results(web_results)
    
    print(f"✓ Document context length: {len(doc_context)} chars")
    print(f"✓ Web context length: {len(web_context)} chars")
    
    assert "Machine learning" in doc_context, "Document content should be in context"
    assert "Latest AI Breakthrough" in web_context, "Web result should be in context"
    print("✅ Context Assembly test passed")


def test_source_citations():
    """Test source citation formatting."""
    print("\n" + "="*60)
    print("TEST 6: Source Citations")
    print("="*60)
    
    from src.models.document import AnswerSource, RetrievedContext
    
    # Create test context
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            parent_doc_id="d1",
            chunk_index=0,
            content="Test content",
            metadata={"source_type": SourceType.PDF, "document_title": "Research Paper"}
        )
    ]
    
    web_results = [
        WebSearchResult(
            query="test",
            title="Web Article",
            url="https://example.com",
            snippet="Web content"
        )
    ]
    
    context = RetrievedContext(
        document_chunks=chunks,
        web_results=web_results,
        query_type=QueryType.HYBRID
    )
    
    sources = context.get_all_sources()
    
    print(f"✓ Retrieved {len(sources)} sources")
    for source in sources:
        print(f"  - {source.format_citation()}")
    
    assert len(sources) == 2, "Should have 2 sources"
    print("✅ Source Citations test passed")


def test_configuration():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TEST 7: Configuration")
    print("="*60)
    
    from src.config import (
        CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K,
        DOCUMENTS_DIR, FAISS_INDEX_DIR
    )
    
    print(f"✓ Chunk size: {CHUNK_SIZE}")
    print(f"✓ Chunk overlap: {CHUNK_OVERLAP}")
    print(f"✓ Default top-k: {DEFAULT_TOP_K}")
    print(f"✓ Documents directory: {DOCUMENTS_DIR}")
    print(f"✓ FAISS index directory: {FAISS_INDEX_DIR}")
    
    assert CHUNK_SIZE > 0, "Chunk size should be positive"
    assert CHUNK_OVERLAP >= 0, "Chunk overlap should be non-negative"
    print("✅ Configuration test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("HYBRID RAG SEARCH ENGINE - COMPONENT TESTS")
    print("="*60)
    
    tests = [
        test_data_models,
        test_text_cleaning,
        test_chunking,
        test_query_classification,
        test_context_assembly,
        test_source_citations,
        test_configuration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Please review the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
