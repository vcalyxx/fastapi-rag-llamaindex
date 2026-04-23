# FastAPI Docs RAG

RAG system built using LlamaIndex + Qdrant + HuggingFace.

## Features

- Loads FastAPI docs
- Chunks text
- Generates embeddings
- Stores in Qdrant
- Answers queries using local LLM

## Setup

```bash
uv sync
docker run -p 6333:6333 qdrant/qdrant
uv run python app.py
```
