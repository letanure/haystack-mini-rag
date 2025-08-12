# Mini RAG System

A lightweight Retrieval-Augmented Generation (RAG) implementation for learning purposes, featuring dense retrieval with Haystack v2 components and OpenAI generation.

## Architecture Flow

```mermaid
graph TD
    A[User Query] --> B{CLI Command}
    
    B -->|minirag ask| C[SimpleRAG.ask()]
    B -->|minirag eval| D[Evaluator.evaluate()]
    B -->|minirag cache| E[Cache Management]
    
    C --> F[1. Setup Phase]
    F --> F1{Cache Check}
    F1 -->|Cache Hit| F2[Load from .cache/]
    F1 -->|Cache Miss| F3[Load docs.jsonl]
    F3 --> F4[Generate Embeddings<br/>SentenceTransformers]
    F4 --> F5[Save to Cache]
    F5 --> F6[Create Document Store]
    F2 --> F6
    F6 --> F7[Store in InMemoryDocumentStore]
    
    C --> G[2. Retrieval Phase]
    G --> G1[Embed Query<br/>Same Model as Docs]
    G1 --> G2[Similarity Search<br/>Cosine Distance]
    G2 --> G3[Return Top-K Documents<br/>with Scores]
    
    C --> H[3. Generation Phase]
    H --> H1[Format Context Prompt<br/>Include Retrieved Docs]
    H1 --> H2[OpenAI API Call<br/>GPT-4o-mini]
    H2 --> H3[Return Generated Answer<br/>with Citations]
    
    D --> I[Evaluation Flow]
    I --> I1[Load Test Cases<br/>golden_test.json 15 cases]
    I1 --> I2[For Each Query]
    I2 --> I3[Search Pipeline<br/>Use SimpleRAG.search()]
    I3 --> I4[Calculate Recall@K<br/>K=1,3,5]
    I2 --> I5[Full Pipeline<br/>Use SimpleRAG.ask()]
    I5 --> I6[Answer Quality<br/>Keyword Overlap vs Expected]
    I4 --> I7[Evaluation Report<br/>Recall + Answer Quality]
    I6 --> I7
    
    E --> E1{Cache Action}
    E1 -->|Info| E2[Show Cache Stats]
    E1 -->|Clear| E3[Delete All Cache]
    
    style F fill:#e1f5fe
    style G fill:#f3e5f5
    style H fill:#e8f5e8
    style I fill:#fff3e0
    style E fill:#fff8e1
```

## Component Overview

```mermaid
graph LR
    subgraph "Data Layer"
        A[docs.jsonl<br/>20 AI snippets]
        B[golden_test.json<br/>15 test cases]
    end
    
    subgraph "Core Components"
        C[SimpleRAG<br/>Main pipeline]
        D[Evaluator<br/>Recall@K + Answer Quality]
        E[EmbeddingCache<br/>Disk-based storage]
    end
    
    subgraph "External Services"
        F[SentenceTransformers<br/>MiniLM embeddings]
        G[OpenAI API<br/>GPT-4o-mini]
    end
    
    subgraph "CLI Interface"
        H[minirag ask<br/>Q&A interface]
        I[minirag eval<br/>Performance testing]
        J[minirag cache<br/>Cache management]
    end
    
    A --> C
    B --> D
    C --> F
    C --> G
    C --> E
    H --> C
    I --> D
    J --> E
    D --> C
    
    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#fff3e0
    style F fill:#e8f5e8
    style G fill:#e8f5e8
```

## Features

- **Dense Retrieval**: Uses Haystack v2 with sentence-transformers (MiniLM) for semantic search
- **OpenAI Generation**: Integrates with GPT-4o-mini for answer generation
- **Smart Caching**: Disk-based embedding cache for 10x faster repeated runs
- **Comprehensive Evaluation**: Recall@K + answer quality metrics with 15 test cases
- **Professional CLI**: Color-coded output with `--refresh-cache`, `--detailed` flags
- **20 AI/ML documents**: Curated knowledge base about AI concepts

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Configure OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Basic Q&A
```bash
minirag ask "What is RAG?"
```

### With source documents
```bash
minirag ask "How do embeddings work?" --show-sources
```

### Adjust retrieval count
```bash
minirag ask "What are transformers?" --k 5 --show-sources
```

### Cache Management
```bash
# First run computes embeddings (~10s)
minirag ask "What is RAG?"

# Subsequent runs use cache (~1s)  
minirag ask "What are embeddings?"

# Force refresh cache
minirag ask "What is BERT?" --refresh-cache

# Manage cache
minirag cache          # Show cache info
minirag cache --clear  # Clear all cached embeddings
```

## Evaluation

Run retrieval evaluation with Recall@K metrics:
```bash
minirag eval
```

Or run directly:
```bash
python -m minirag.cli ask "What is RAG?"
python -m minirag.cli eval
```

Output shows:
- Recall@1: 80% (4/5 queries get the right doc first)
- Recall@3: 100% (all queries find relevant docs in top 3)
- Recall@5: 100% (perfect recall at k=5)

## Example Output

```
Question: What is RAG?

Top 4 Retrieved Documents:
────────────────────────────────────────
  1. [id=1] (score: 0.367)
     Retrieval-Augmented Generation (RAG) combines a retriever with a generator...
  
Generating answer...

Answer:
────────────────────────────────────────
RAG stands for Retrieval-Augmented Generation, which combines a retriever 
with a generator to produce grounded answers [1].
────────────────────────────────────────
```

## Project Structure

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

## Technical Details

- **Embeddings**: 384-dimensional vectors from `all-MiniLM-L6-v2`
- **Retrieval**: Cosine similarity on normalized vectors
- **Generation**: OpenAI ChatCompletions API with citation prompting
- **Code**: ~150 lines total, clean and readable

## Learning Roadmap

Ready to level up? Here's a suggested learning path with increasing complexity:

### 🟢 **Beginner Extensions** (Start Here)
1. **Add More Document Types**: Load PDFs, Word docs, web scraping
2. **Improve Evaluation**: Add more test queries, measure answer quality
3. **Simple Caching**: Cache embeddings to disk, avoid re-computing
4. **Basic Error Handling**: Retry failed API calls, better error messages

### 🟡 **Intermediate RAG** (Build Production Skills)  
5. **Hybrid Search**: Combine semantic search + keyword (BM25) search
6. **Document Chunking**: Split long docs into smaller, searchable pieces
7. **Multiple Models**: Compare different embedding models (OpenAI, Cohere)
8. **Streaming Responses**: Stream OpenAI answers for better UX

### 🔴 **Advanced RAG** (Research-Level Techniques)
9. **Re-ranking**: Use cross-encoder models to re-rank retrieved docs
10. **Query Expansion**: Generate multiple query variations for better recall
11. **Vector Databases**: Replace in-memory with Pinecone/Weaviate/Chroma
12. **Agent Workflows**: Multi-step reasoning, tool use, function calling

### 🚀 **Production RAG** (Real-World Deployment)
13. **Evaluation Suite**: Automated testing, A/B testing, user feedback
14. **Monitoring**: Track performance, costs, user satisfaction  
15. **Security**: Input sanitization, rate limiting, content filtering
16. **Scalability**: Load balancing, distributed embeddings, cost optimization

**Next Suggested Step**: Pick one beginner extension and implement it! Start with #2 (better evaluation) to measure improvements.
