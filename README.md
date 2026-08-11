# Document QA Bot

A lightweight RAG (Retrieval-Augmented Generation) system that answers questions
about an uploaded PDF, grounded strictly in the document's content. Runs entirely
locally using Ollama — no API keys, no external services.

## How it works

1. **Extract** — pulls raw text out of the uploaded PDF (`pypdf`)
2. **Chunk** — splits the text into overlapping segments for precise retrieval (`langchain-text-splitters`)
3. **Embed** — converts each chunk into a vector using a local embedding model (`nomic-embed-text` via Ollama)
4. **Store** — persists chunk vectors in a local vector database (`ChromaDB`)
5. **Retrieve** — on a question, finds the most semantically relevant chunks
6. **Generate** — feeds retrieved chunks + question to a local LLM (`llama3.1` via Ollama), instructed to answer only from that context

If the answer isn't in the document, the bot says so instead of guessing.

## Tech stack

- Python
- Streamlit (UI)
- ChromaDB (vector store)
- Ollama (local embeddings + LLM — `nomic-embed-text`, `llama3.1`)
- LangChain text splitters

## Setup

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
4. Upload a PDF, click "Process document," and ask questions.

## Features

- Fully local — no API keys or per-query costs
- Answers grounded strictly in the uploaded document
- Shows the exact source chunks used for each answer
- Handles unreadable/scanned PDFs and offline-LLM errors gracefully

## Possible extensions

- Multi-document support
- Conversation memory for follow-up questions
- Swap Ollama for OpenAI/GPT-4o-mini via a config flag