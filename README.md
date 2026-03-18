# Multi-Document-RAG-Search-Engine

A powerful multi-document RAG (Retrieval-Augmented Generation) search engine that combines **local document search** with **real-time web search** to provide comprehensive, cited answers to user queries.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Multi-Source Retrieval**: Search across PDFs, text files, Wikipedia pages, and real-time web results
- **Intelligent Query Routing**: Automatically classifies queries as document-based, web-based, or hybrid
- **Citation-Aware Answers**: All responses include source citations ([Doc] or [Web])
- **Real-Time Web Search**: Powered by Tavily for up-to-date information
- **Dual LLM Support**: Works with both OpenAI and Google Gemini
- **Clean Streamlit UI**: Interactive chat interface with evidence tabs

## Architecture

```
User Query → Query Classification → Retrieval Strategy → Context Assembly → Answer Generation
                ↓                        ↓                    ↓                  ↓
           (LLM-based)           (FAISS / Tavily)      (Merge sources)    (Cited response)
```

## Tech Stack

- **LLM Orchestration**: LangChain
- **LLM Providers**: OpenAI GPT-4o-mini / Google Gemini 2.5 Flash
- **Vector Database**: FAISS with OpenAI embeddings
- **Web Search**: Tavily API
- **Document Processing**: PyPDF, Wikipedia API
- **Frontend**: Streamlit
- **Embeddings**: OpenAI text-embedding-3-small

## Installation

### Prerequisites

- Python 3.11+
- API Keys:
  - [Google AI Studio](https://aistudio.google.com/app/apikey) (for Gemini)
  - [Tavily](https://tavily.com/) (for web search)
  - [OpenAI](https://platform.openai.com/) (optional, for embeddings)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hybrid-rag-search-engine.git
   cd hybrid-rag-search-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   
   Create a `.env` file:
   ```env
   # Required for LLM (choose one)
   GOOGLE_API_KEY=your_google_api_key
   USE_GOOGLE_LLM=true
   GOOGLE_LLM_MODEL=gemini-2.5-flash

   # Required for web search
   TAVILY_API_KEY=your_tavily_api_key

   # Optional (for document embeddings)
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Run the Application

```bash
python main.py
```

Or directly with Streamlit:
```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

### Using the Interface

1. **Upload Documents** (requires OpenAI key):
   - Drag and drop PDF, TXT, or MD files
   - Or import Wikipedia pages directly

2. **Ask Questions**:
   - Type any question in the chat
   - The system auto-classifies and routes to appropriate sources
   - View answers with source citations

3. **Explore Evidence**:
   - **Document Evidence**: Retrieved chunks from your documents
   - **Web Evidence**: Raw web search results
   - **Sources**: Formatted citation list

## Query Types

The system automatically classifies queries:

| Query Type | Example | Sources Used |
|------------|---------|--------------|
| **Document** | "Explain attention mechanism" | FAISS index only |
| **Web** | "Latest AI news today" | Tavily web search only |
| **Hybrid** | "How does RAG compare with current tools?" | Both FAISS + Tavily |

## Project Structure

```
hybrid-rag-search-engine/
├── app/
│   └── streamlit_app.py          # Streamlit UI
├── src/
│   ├── models/
│   │   ├── document.py           # Data schemas
│   │   └── query.py              # Query models
│   ├── ingestion/
│   │   ├── loaders.py            # PDF, Wikipedia, text loaders
│   │   ├── cleaning.py           # Text normalization
│   │   └── chunking.py           # Document chunking
│   ├── vectorstore/
│   │   ├── embeddings.py         # Embedding configuration
│   │   └── faiss_store.py        # FAISS index management
│   ├── retrieval/
│   │   ├── query_router.py       # Query classification
│   │   ├── web_search.py         # Tavily integration
│   │   └── context_assembly.py   # Context building
│   ├── generation/
│   │   ├── answer_generator.py   # Answer generation with citations
│   │   └── summarizer.py         # Document summarization
│   └── config.py                 # Configuration
├── data/
│   ├── documents/                # Upload location
│   └── faiss_index/              # Persisted index
├── requirements.txt
├── main.py                       # Entry point
└── test_rag.py                   # Component tests
```

## Testing

Run component tests:
```bash
python test_rag.py
```

Tests cover:
- Data models
- Text cleaning
- Document chunking
- Query classification
- Context assembly
- Source citations

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `TAVILY_API_KEY` | Tavily web search API key | - |
| `OPENAI_API_KEY` | OpenAI API key (for embeddings) | - |
| `USE_GOOGLE_LLM` | Use Gemini instead of OpenAI | `false` |
| `GOOGLE_LLM_MODEL` | Gemini model name | `gemini-2.5-flash` |
| `TAVILY_MAX_RESULTS` | Max web search results | `5` |

## Limitations

- **Document search requires OpenAI**: FAISS embeddings use OpenAI by default
- **Rate limits**: Free tiers have usage limits (automatic retry implemented)
- **Context size**: Large documents are chunked to fit token limits

## Future Enhancements

- [ ] Multi-modal support (images, audio)
- [ ] Conversation memory
- [ ] User authentication
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Alternative embedding providers (HuggingFace)
- [ ] Document summarization dashboard

## License

MIT License - feel free to use and modify!

## Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Web search powered by [Tavily](https://tavily.com/)
- LLM by [Google Gemini](https://deepmind.google/technologies/gemini/)

---
