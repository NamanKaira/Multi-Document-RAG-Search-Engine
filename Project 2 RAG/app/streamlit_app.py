"""
Streamlit UI for the Hybrid RAG Search Engine.
"""

import os
import time
from pathlib import Path

import streamlit as st

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DOCUMENTS_DIR, FAISS_INDEX_DIR, validate_config,
    OPENAI_API_KEY, GOOGLE_API_KEY, TAVILY_API_KEY, USE_GOOGLE_LLM
)
from src.ingestion.loaders import load_documents, get_document_stats
from src.ingestion.chunking import chunk_documents, get_chunk_statistics
from src.vectorstore.faiss_store import FAISSDocumentStore, load_faiss_index
from src.retrieval.query_router import classify_query, should_search_web, should_search_documents
from src.retrieval.web_search import WebSearchClient
from src.models.document import RetrievedContext, QueryType
from src.generation.answer_generator import AnswerGenerator
from src.generation.summarizer import get_top_document_summaries, format_summaries


# Page configuration
st.set_page_config(
    page_title="Hybrid RAG Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_store" not in st.session_state:
        st.session_state.document_store = None
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True
    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None


def get_source_icon(query_type: QueryType) -> str:
    """Get icon for query type."""
    icons = {
        QueryType.DOCUMENT: "📄",
        QueryType.WEB: "🌐",
        QueryType.HYBRID: "🔀"
    }
    return icons.get(query_type, "📄")


def get_source_label(query_type: QueryType) -> str:
    """Get label for query type."""
    labels = {
        QueryType.DOCUMENT: "Document-based",
        QueryType.WEB: "Web-based",
        QueryType.HYBRID: "Hybrid"
    }
    return labels.get(query_type, "Document-based")


def render_sidebar():
    """Render the sidebar with document management."""
    with st.sidebar:
        st.title("📚 Document Manager")
        
        # API Status
        st.subheader("API Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            if OPENAI_API_KEY:
                st.success("OpenAI ✓")
            else:
                st.warning("OpenAI ✗")
        with col2:
            if GOOGLE_API_KEY:
                st.success("Google ✓")
            else:
                st.warning("Google ✗")
        with col3:
            if TAVILY_API_KEY:
                st.success("Tavily ✓")
            else:
                st.error("Tavily ✗")
        
        # Show active LLM provider
        if USE_GOOGLE_LLM and GOOGLE_API_KEY:
            st.info("Using Google Gemini for LLM")
        elif OPENAI_API_KEY:
            st.info("Using OpenAI for LLM")
        
        st.divider()
        
        # Web Search Toggle
        st.subheader("Search Settings")
        st.session_state.web_search_enabled = st.toggle(
            "Enable Web Search (Tavily)",
            value=st.session_state.web_search_enabled,
            help="Enable real-time web search for queries requiring current information"
        )
        
        st.divider()
        
        # File Upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process & Index Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    process_uploaded_files(uploaded_files)
        
        st.divider()
        
        # Wikipedia Import
        st.subheader("Import from Wikipedia")
        wiki_query = st.text_input("Search Wikipedia", placeholder="e.g., Artificial intelligence")
        if wiki_query and st.button("Import Wikipedia Page"):
            with st.spinner("Loading Wikipedia page..."):
                process_wikipedia(wiki_query)
        
        st.divider()
        
        # Indexed Files
        st.subheader("Indexed Documents")
        if st.session_state.indexed_files:
            for fname in st.session_state.indexed_files:
                st.text(f"📄 {fname}")
            
            if st.button("Clear Index", type="secondary"):
                clear_index()
        else:
            st.info("No documents indexed yet")
        
        # Index stats
        if st.session_state.document_store:
            try:
                count = st.session_state.document_store.get_document_count()
                st.caption(f"Total chunks in index: {count}")
            except:
                pass


def process_uploaded_files(uploaded_files):
    """Process and index uploaded files."""
    saved_paths = []
    
    for uploaded_file in uploaded_files:
        # Save to documents directory
        save_path = DOCUMENTS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        saved_paths.append(save_path)
        st.session_state.indexed_files.append(uploaded_file.name)
    
    # Load and process documents
    documents = load_documents(saved_paths)
    
    if not documents:
        st.error("No documents could be loaded")
        return
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    # Create or update index
    if st.session_state.document_store is None:
        st.session_state.document_store = FAISSDocumentStore()
        st.session_state.document_store.create_index(chunks)
    else:
        st.session_state.document_store.add_documents(chunks)
    
    # Save index
    st.session_state.document_store.save()
    
    stats = get_chunk_statistics(chunks)
    st.success(f"Indexed {stats['total_chunks']} chunks from {len(documents)} documents")


def process_wikipedia(query: str):
    """Process Wikipedia page."""
    documents = load_documents([{"type": "wikipedia", "query": query}])
    
    if not documents:
        st.error(f"Could not load Wikipedia page for '{query}'")
        return
    
    chunks = chunk_documents(documents)
    
    if st.session_state.document_store is None:
        st.session_state.document_store = FAISSDocumentStore()
        st.session_state.document_store.create_index(chunks)
    else:
        st.session_state.document_store.add_documents(chunks)
    
    st.session_state.document_store.save()
    st.session_state.indexed_files.append(f"Wikipedia: {query}")
    
    st.success(f"Indexed Wikipedia page: {query}")


def clear_index():
    """Clear the document index."""
    st.session_state.document_store = None
    st.session_state.indexed_files = []
    
    # Remove saved index files
    import shutil
    if FAISS_INDEX_DIR.exists():
        shutil.rmtree(FAISS_INDEX_DIR)
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    st.success("Index cleared")


def process_query(query: str) -> dict:
    """Process a user query through the RAG pipeline."""
    start_time = time.time()
    
    # Classify query
    classification = classify_query(query)
    
    # Override if web search disabled
    if not st.session_state.web_search_enabled:
        classification.query_type = QueryType.DOCUMENT
    
    # Initialize context
    doc_chunks = []
    web_results = []
    
    # Search documents if needed
    if should_search_documents(classification) and st.session_state.document_store:
        try:
            doc_results = st.session_state.document_store.search(query, top_k=5)
            doc_chunks = [chunk for chunk, score in doc_results]
        except Exception as e:
            st.warning(f"Document search error: {e}")
    
    # Search web if needed
    if should_search_web(classification) and st.session_state.web_search_enabled:
        try:
            client = WebSearchClient()
            web_results = client.search(query)
        except Exception as e:
            st.warning(f"Web search error: {e}")
    
    # Create context
    context = RetrievedContext(
        document_chunks=doc_chunks,
        web_results=web_results,
        query_type=classification.query_type
    )
    
    # Generate answer
    try:
        generator = AnswerGenerator()
        result = generator.generate(query, context)
        result.processing_time_ms = (time.time() - start_time) * 1000
    except Exception as e:
        result = type('obj', (object,), {
            'query': query,
            'answer': f"Error generating answer: {e}",
            'sources': [],
            'context': context,
            'query_type': classification.query_type,
            'processing_time_ms': (time.time() - start_time) * 1000
        })()
    
    return result


def render_chat_interface():
    """Render the main chat interface."""
    st.title("🔍 Hybrid RAG Search Engine")
    st.caption("Ask questions about your documents or get real-time information from the web")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show source info for assistant messages
            if message["role"] == "assistant" and "query_type" in message:
                icon = get_source_icon(message["query_type"])
                label = get_source_label(message["query_type"])
                st.caption(f"{icon} {label} answer")
    
    # Chat input
    if query := st.chat_input("Ask a question..."):
        # Validate we have something to search
        if st.session_state.document_store is None and not st.session_state.web_search_enabled:
            st.error("Please upload documents or enable web search")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                result = process_query(query)
                st.session_state.last_query_result = result
            
            # Display answer
            st.markdown(result.answer)
            
            # Show source indicator
            icon = get_source_icon(result.query_type)
            label = get_source_label(result.query_type)
            st.caption(f"{icon} {label} answer | Processed in {result.processing_time_ms:.0f}ms")
        
        # Add to message history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "query_type": result.query_type
        })


def render_evidence_tabs():
    """Render evidence tabs below chat."""
    if st.session_state.last_query_result is None:
        return
    
    result = st.session_state.last_query_result
    
    tab1, tab2, tab3 = st.tabs(["📄 Document Evidence", "🌐 Web Evidence", "📊 Sources"])
    
    with tab1:
        if result.context and result.context.document_chunks:
            # Get document summaries
            summaries = get_top_document_summaries(result.context.document_chunks, top_n=3)
            st.markdown(format_summaries(summaries))
            
            st.divider()
            st.subheader("Retrieved Chunks")
            
            for i, chunk in enumerate(result.context.document_chunks, 1):
                with st.expander(f"Chunk {i}: {chunk.document_title}"):
                    st.text(f"Index: {chunk.chunk_index + 1}")
                    st.markdown(chunk.content)
        else:
            st.info("No document evidence retrieved")
    
    with tab2:
        if result.context and result.context.web_results:
            for i, web_result in enumerate(result.context.web_results, 1):
                with st.expander(f"{i}. {web_result.title}"):
                    st.markdown(f"**URL:** [{web_result.url}]({web_result.url})")
                    st.markdown(f"**Content:**\n{web_result.content or web_result.snippet}")
        else:
            st.info("No web evidence retrieved")
    
    with tab3:
        if result.sources:
            for i, source in enumerate(result.sources, 1):
                st.markdown(f"{i}. {source.format_citation()}")
        else:
            st.info("No sources available")


def main():
    """Main application entry point."""
    init_session_state()
    
    # Check configuration
    errors = validate_config()
    if errors:
        st.warning("⚠️ Configuration Issues:\n" + "\n".join(errors))
    
    render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat_interface()
    
    with col2:
        st.subheader("Evidence & Sources")
        render_evidence_tabs()


if __name__ == "__main__":
    main()
