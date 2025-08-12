# Mini RAG System 🤖

A lightweight Retrieval-Augmented Generation (RAG) implementation for learning purposes, featuring dense retrieval with sentence-transformers and OpenAI generation.

## Features ✨

- 🔍 **Dense Retrieval**: Uses sentence-transformers (MiniLM) for semantic search
- 💬 **OpenAI Generation**: Integrates with GPT-4o-mini for answer generation
- 📊 **Evaluation**: Recall@K metrics for retrieval quality assessment
- 🎨 **Beautiful CLI**: Color-coded output with progress indicators
- 📚 **20 AI/ML documents**: Curated knowledge base about AI concepts

## Setup 🚀

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage 💻

### Basic Q&A
```bash
python app.py "What is RAG?"
```

### With source documents
```bash
python app.py "How do embeddings work?" --show-sources
```

### Adjust retrieval count
```bash
python app.py "What are transformers?" --k 5 --show-sources
```

## Evaluation 📈

Run retrieval evaluation with Recall@K metrics:
```bash
python eval.py
```

Output shows:
- Recall@1: 80% (4/5 queries get the right doc first)
- Recall@3: 100% (all queries find relevant docs in top 3)
- Recall@5: 100% (perfect recall at k=5)

## Example Output 🎯

```
❓ Question: What is RAG?

🔍 Top 4 Retrieved Documents:
────────────────────────────────────────
  1. [id=1] (score: 0.367)
     Retrieval-Augmented Generation (RAG) combines a retriever with a generator...
  
💡 Generating answer...

✨ Answer:
────────────────────────────────────────
RAG stands for Retrieval-Augmented Generation, which combines a retriever 
with a generator to produce grounded answers [1].
────────────────────────────────────────
```

## Project Structure 📁

```
.
├── app.py              # Main RAG application
├── eval.py             # Evaluation script
├── data/
│   ├── docs.jsonl      # Document collection (20 AI snippets)
│   └── golden_test.json # Test queries with ground truth
├── requirements.txt    # Python dependencies
└── .env               # API keys (create from .env.example)
```

## Technical Details 🔧

- **Embeddings**: 384-dimensional vectors from `all-MiniLM-L6-v2`
- **Retrieval**: Cosine similarity on normalized vectors
- **Generation**: OpenAI ChatCompletions API with citation prompting
- **Code**: ~150 lines total, clean and readable

## What I'd Improve Next 🚧

1. **Hybrid Retrieval**: Combine dense + sparse (BM25) for better recall
2. **Vector Database**: Use Pinecone/Weaviate for production scale
3. **Better Evaluation**: Add semantic similarity and answer quality metrics
4. **Streaming**: Stream OpenAI responses for better UX
5. **Caching**: Cache embeddings and common queries
6. **Error Recovery**: Retry logic for API failures
7. **Haystack v2 Pipeline**: Leverage full Haystack framework capabilities
