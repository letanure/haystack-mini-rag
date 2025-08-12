"""Data models for the RAG system."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """Search result with similarity score."""

    id: str
    content: str
    score: Optional[float] = None


@dataclass
class EvalResults:
    """Evaluation results for retrieval and generation."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    answer_relevance: float
    total_queries: int
