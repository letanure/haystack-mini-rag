"""Core RAG functionality - combines retrieval + generation for grounded answers."""

import os
import json
from dataclasses import dataclass
from typing import List

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from .cache import EmbeddingCache

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class SearchResult:
    """Search result with similarity score."""
    id: str
    content: str
    score: float = None


class SimpleRAG:
    """Simple RAG system following the load -> embed -> search -> generate pipeline."""
    
    def __init__(self, docs_path: str = "data/docs.jsonl", 
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_cache: bool = True):
        self.docs_path = docs_path
        self.embed_model = embed_model  # MiniLM: fast, 384 dimensions, good quality
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None
        self.retriever = None
        self.query_embedder = None
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent fork warnings
    
    def load_documents(self) -> List[Document]:
        """Load documents from JSONL file."""
        if not os.path.exists(self.docs_path):
            raise FileNotFoundError(f"Documents file not found: {self.docs_path}")
            
        docs = []
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    if "id" not in data or "content" not in data:
                        raise ValueError(f"Missing 'id' or 'content' in line {line_num}")
                    docs.append(Document(id=data["id"], content=data["content"]))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in line {line_num}: {e}")
        
        if not docs:
            raise ValueError(f"No valid documents found in {self.docs_path}")
        return docs
    
    def setup(self, force_refresh: bool = False):
        """Initialize the RAG system - embed documents once, query many times."""
        store = InMemoryDocumentStore()
        embedded_docs = None
        
        # Try to load from cache first
        if self.use_cache and not force_refresh:
            embedded_docs = self.cache.get_cached_docs(self.docs_path, self.embed_model)
        
        if embedded_docs is not None:
            print("📦 Using cached embeddings")
            store.write_documents(embedded_docs)
        else:
            print("🔄 Computing embeddings...")
            docs = self.load_documents()
            
            # Convert documents to embeddings
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model=self.embed_model, progress_bar=False
            )
            doc_embedder.warm_up()
            embedded_result = doc_embedder.run(documents=docs)
            embedded_docs = embedded_result["documents"]
            
            # Cache the embedded documents
            if self.use_cache:
                self.cache.save_embedded_docs(self.docs_path, self.embed_model, embedded_docs)
                print("💾 Cached embeddings for future use")
            
            store.write_documents(embedded_docs)
        
        # Create retriever and query embedder (same model for consistency)
        self.retriever = InMemoryEmbeddingRetriever(document_store=store)
        self.query_embedder = SentenceTransformersTextEmbedder(
            model=self.embed_model, progress_bar=False
        )
        self.query_embedder.warm_up()
    
    def search(self, query: str, k: int = 4) -> List[SearchResult]:
        """Find most similar documents using semantic search."""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        if not self.retriever:
            raise RuntimeError("RAG system not initialized. Call setup() first.")
            
        query_emb = self.query_embedder.run(text=query)
        results = self.retriever.run(query_embedding=query_emb["embedding"], top_k=k)
        
        return [
            SearchResult(id=doc.id, content=doc.content, score=getattr(doc, "score", None))
            for doc in results["documents"]
        ]
    
    def generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        """Generate answer from context using OpenAI."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not sources:
            raise ValueError("No sources provided for generation")
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ OPENAI_API_KEY not set in .env file"
        
        if not OpenAI:
            return "❌ openai package not installed"
        
        # Format sources as context with citations
        context = "\n".join([f"[{s.id}] {s.content}" for s in sources])
        
        prompt = f"""Answer based on this context only. Cite sources with [id].

Question: {question}

Context:
{context}

Answer:"""
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful, precise assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent answers
        )
        
        return response.choices[0].message.content.strip()
    
    def ask(self, question: str, k: int = 4) -> tuple[str, List[SearchResult]]:
        """Complete RAG pipeline: retrieve -> generate."""
        sources = self.search(question, k)
        answer = self.generate_answer(question, sources)
        return answer, sources