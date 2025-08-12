Tiny RAG demo (Haystack v2) with:

- In-memory store
- Dense retriever (MiniLM)
- OpenAI generator
- CLI for Q&A
- Simple Recall@K eval

## Setup

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Set your OPENAI_API_KEY
