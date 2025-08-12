"""Core RAG functionality."""

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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class SearchResult:
    """A search result."""
    id: str
    content: str
    score: float = None


class SimpleRAG:
    """Simple RAG system using Haystack v2."""
    
    def __init__(self, docs_path: str = "data/docs.jsonl", 
                 embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.docs_path = docs_path
        self.embed_model = embed_model
        self.retriever = None
        self.query_embedder = None
        
        # Suppress warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def load_documents(self) -> List[Document]:
        """Load documents from JSONL file."""
        docs = []
        with open(self.docs_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                docs.append(Document(id=data["id"], content=data["content"]))
        return docs
    
    def setup(self):
        """Initialize the RAG system."""
        docs = self.load_documents()
        
        # Create store and embed documents
        store = InMemoryDocumentStore()
        doc_embedder = SentenceTransformersDocumentEmbedder(
            model=self.embed_model, progress_bar=False
        )
        doc_embedder.warm_up()
        embedded_docs = doc_embedder.run(documents=docs)
        store.write_documents(embedded_docs["documents"])
        
        # Create retriever and query embedder
        self.retriever = InMemoryEmbeddingRetriever(document_store=store)
        self.query_embedder = SentenceTransformersTextEmbedder(
            model=self.embed_model, progress_bar=False
        )
        self.query_embedder.warm_up()
    
    def search(self, query: str, k: int = 4) -> List[SearchResult]:
        """Search for relevant documents."""
        query_emb = self.query_embedder.run(text=query)
        results = self.retriever.run(query_embedding=query_emb["embedding"], top_k=k)
        
        return [
            SearchResult(id=doc.id, content=doc.content, score=getattr(doc, "score", None))
            for doc in results["documents"]
        ]
    
    def generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        """Generate answer using OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ OPENAI_API_KEY not set in .env file"
        
        if not OpenAI:
            return "❌ openai package not installed"
        
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
            temperature=0.1,
        )
        
        return response.choices[0].message.content.strip()
    
    def ask(self, question: str, k: int = 4) -> tuple[str, List[SearchResult]]:
        """Ask a question and get answer with sources."""
        sources = self.search(question, k)
        answer = self.generate_answer(question, sources)
        return answer, sources