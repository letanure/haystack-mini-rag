"""
Core RAG (Retrieval-Augmented Generation) functionality.

RAG combines two key ideas:
1. Retrieval: Find relevant documents using semantic similarity
2. Generation: Use those documents as context for an LLM to generate answers

This keeps answers grounded in facts rather than hallucinated.
"""

import os
import json
from dataclasses import dataclass
from typing import List

# Haystack v2 components - why we use these:
# - Professional, well-tested RAG components
# - Easy to swap storage backends later (Pinecone, etc.)
# - Handles embedding complexities for us
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

# Optional import pattern - graceful degradation if OpenAI not available
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class SearchResult:
    """Container for search results - cleaner than raw dictionaries."""
    id: str
    content: str
    score: float = None  # Similarity score (higher = more relevant)


class SimpleRAG:
    """
    Simple RAG system - learns the core concepts without complexity.
    
    Why this architecture:
    - Single class keeps related functionality together
    - Separate setup() lets you control initialization timing
    - Methods follow the RAG pipeline: load -> embed -> search -> generate
    """
    
    def __init__(self, docs_path: str = "data/docs.jsonl", 
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.docs_path = docs_path
        self.embed_model = embed_model  # MiniLM is fast, good quality, 384 dimensions
        self.retriever = None
        self.query_embedder = None
        
        # Prevents fork warnings - not important for learning, just cleanup
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def load_documents(self) -> List[Document]:
        """Load documents from JSONL file.
        
        Why JSONL? Each line is a separate JSON object - easy to stream large files.
        Alternative: Could use CSV, but JSON handles text with quotes/newlines better.
        """
        docs = []
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Haystack Document objects - standardized format for RAG
                docs.append(Document(id=data["id"], content=data["content"]))
        return docs
    
    def setup(self):
        """Initialize the RAG system - the "indexing" phase.
        
        Why separate setup? Embedding is expensive - do it once, query many times.
        In production, you'd pre-compute embeddings and save them.
        """
        docs = self.load_documents()
        
        # Step 1: Create document store (in-memory for simplicity)
        # Production: Use Pinecone, Weaviate, or other vector DB
        store = InMemoryDocumentStore()
        
        # Step 2: Convert documents to embeddings (vectors)
        # Why embeddings? Numbers capture semantic meaning - "king" + "woman" ≈ "queen"
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model=self.embed_model, progress_bar=False
        )
        doc_embedder.warm_up()  # Downloads model if first time
        embedded_docs = doc_embedder.run(documents=docs)
        store.write_documents(embedded_docs["documents"])
        
        # Step 3: Create retriever and query embedder
        # Same model for queries and docs - ensures they're in same vector space
        self.retriever = InMemoryEmbeddingRetriever(document_store=store)
        self.query_embedder = SentenceTransformersTextEmbedder(
            model=self.embed_model, progress_bar=False
        )
        self.query_embedder.warm_up()
    
    def search(self, query: str, k: int = 4) -> List[SearchResult]:
        """Search for relevant documents using semantic similarity.
        
        Why semantic search works:
        - "What is RAG?" and "Retrieval Augmented Generation" are similar in vector space
        - Much better than keyword matching for questions
        """
        # Convert query to same vector space as documents
        query_emb = self.query_embedder.run(text=query)
        
        # Find most similar documents using cosine similarity
        # Higher scores = more similar to query
        results = self.retriever.run(query_embedding=query_emb["embedding"], top_k=k)
        
        # Convert to our simple format
        return [
            SearchResult(id=doc.id, content=doc.content, score=getattr(doc, "score", None))
            for doc in results["documents"]
        ]
    
    def generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        """Generate answer using OpenAI - the "generation" part of RAG.
        
        Why this approach:
        - Provide context to LLM so it doesn't hallucinate
        - Ask for source citations for transparency
        - Low temperature (0.1) for consistent, factual answers
        """
        # Graceful error handling
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ OPENAI_API_KEY not set in .env file"
        
        if not OpenAI:
            return "❌ openai package not installed"
        
        # Format retrieved documents as context
        # Why format with [id]? Makes it easy for LLM to cite sources
        context = "\n".join([f"[{s.id}] {s.content}" for s in sources])
        
        # Prompt engineering: Be explicit about what you want
        prompt = f"""Answer based on this context only. Cite sources with [id].

Question: {question}

Context:
{context}

Answer:"""
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast, cheap, good enough for most tasks
            messages=[
                {"role": "system", "content": "You are a helpful, precise assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low randomness for factual answers
        )
        
        return response.choices[0].message.content.strip()
    
    def ask(self, question: str, k: int = 4) -> tuple[str, List[SearchResult]]:
        """Complete RAG pipeline: retrieve relevant docs, then generate answer.
        
        Why this flow:
        1. Search finds relevant context (retrieval)
        2. Generate uses that context to create grounded answers (augmented generation)
        
        Returns both answer and sources for transparency.
        """
        sources = self.search(question, k)  # Retrieval step
        answer = self.generate_answer(question, sources)  # Generation step
        return answer, sources