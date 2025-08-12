"""Mini RAG - A lightweight Retrieval-Augmented Generation system."""

__version__ = "0.1.0"

from .cache import EmbeddingCache
from .evaluator import Evaluator
from .loaders import DocumentLoader
from .models import EvalResults, SearchResult
from .rag import SimpleRAG

__all__ = [
    "SimpleRAG",
    "SearchResult",
    "EvalResults",
    "Evaluator",
    "EmbeddingCache",
    "DocumentLoader",
]
