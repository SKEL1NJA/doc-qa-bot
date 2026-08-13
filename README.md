# Document QA Bot

A lightweight RAG (Retrieval-Augmented Generation) system that answers questions
about an uploaded PDF, grounded strictly in the document's content.

Supports two modes: fully local (via Ollama, no API key, no cost) for development,
or Gemini API (free tier, no credit card required) for deployment.

**Live demo:** https://pdf-question-answer-bot.streamlit.app/

## How it works

1. **Extract** — pulls raw text out of the uploaded PDF (`pypdf`)
2. **Chunk** — splits the text into overlapping segments for precise retrieval (`langchain-text-splitters`)
3. **Embed** — converts each chunk into a vector using an embedding model
4. **Store** — persists chunk vectors in a local vector database (`ChromaDB`)
5. **Retrieve** — on a question, finds the most semantically relevant chunks
6. **Generate** — feeds retrieved chunks + question to an LLM, instructed to answer only from that context

If the answer isn't in the document, the bot says so instead of guessing.

## Tech stack

- Python
- Streamlit (UI)
- ChromaDB (vector store)
- LangChain text splitters
- **LLM/embedding provider (switchable):**
  - Ollama — local, free, offline (`nomic-embed-text`, `llama3.1`)
  - Gemini API — free tier, no credit card (`gemini-embedding-001`, `gemini-3.6-flash`)

## Setup

### Option A: Run locally with Ollama (no API key)

1. Install [Ollama](https://ollama.com/download) and pull the required models:
   ```
   ollama pull nomic-embed-text
   ollama pull llama3.1
   ```
2. Clone this repo and set up the environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run src\app.py
   ```

### Option B: Run locally with Gemini (mirrors deployed behavior)

1. Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Set environment variables, then run:
   ```
   $env:LLM_PROVIDER = "gemini"
   $env:GEMINI_API_KEY = "your-key-here"
   streamlit run src\app.py
   ```

Either way: upload a PDF, click "Process document," and ask questions.

## Deployment

Deployed on Streamlit Community Cloud using the Gemini provider (Ollama can't run
on most free hosting platforms). `LLM_PROVIDER` and `GEMINI_API_KEY` are set as
app secrets, not committed to the repo.

## Features

- Switchable local/hosted LLM backend via a single environment variable
- Answers grounded strictly in the uploaded document
- Shows the exact source chunks used for each answer
- Handles unreadable/scanned PDFs and provider errors gracefully

## Possible extensions

- Multi-document support
- Conversation memory for follow-up questions
- Reranking retrieved chunks for higher precision