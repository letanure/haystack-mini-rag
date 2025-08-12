"""Simple tests for RAG system."""

import os
from minirag.rag import SimpleRAG


def test_load_documents():
    """Test document loading."""
    if os.path.exists("data/docs.jsonl"):
        rag = SimpleRAG()
        docs = rag.load_documents()
        assert len(docs) > 0
        assert all(doc.id and doc.content for doc in docs)


def test_search():
    """Test search functionality."""
    if os.path.exists("data/docs.jsonl"):
        rag = SimpleRAG()
        rag.setup()
        
        results = rag.search("What is RAG?", k=3)
        assert len(results) <= 3
        assert all(r.id and r.content for r in results)