"""Enhanced evaluation with answer quality metrics."""

import json
from typing import List, Dict
from dataclasses import dataclass

from .rag import SimpleRAG


@dataclass
class EvalResults:
    """Evaluation results for retrieval and generation."""
    recall_at_1: float
    recall_at_3: float  
    recall_at_5: float
    answer_relevance: float
    total_queries: int


class Evaluator:
    """Enhanced evaluator for RAG systems."""
    
    def __init__(self, rag_system: SimpleRAG):
        self.rag = rag_system
    
    def evaluate(self, test_file: str) -> EvalResults:
        """Run comprehensive evaluation."""
        with open(test_file) as f:
            test_data = json.load(f)["test_cases"]
        
        # Retrieval metrics
        recall_1 = self._recall_at_k(test_data, 1)
        recall_3 = self._recall_at_k(test_data, 3) 
        recall_5 = self._recall_at_k(test_data, 5)
        
        # Answer quality metric (simple keyword overlap)
        relevance = self._answer_relevance(test_data)
        
        return EvalResults(
            recall_at_1=recall_1,
            recall_at_3=recall_3,
            recall_at_5=recall_5,
            answer_relevance=relevance,
            total_queries=len(test_data)
        )
    
    def _recall_at_k(self, test_data: List[Dict], k: int) -> float:
        """Calculate Recall@K for retrieval."""
        hits = 0
        for case in test_data:
            sources = self.rag.search(case["query"], k)
            returned_ids = {s.id for s in sources}
            relevant_ids = set(case["relevant_doc_ids"])
            if returned_ids & relevant_ids:
                hits += 1
        return hits / len(test_data)
    
    def _answer_relevance(self, test_data: List[Dict]) -> float:
        """Simple answer quality metric using keyword overlap."""
        total_score = 0
        evaluated = 0
        
        for case in test_data:
            if "expected_answer" not in case:
                continue
                
            answer, _ = self.rag.ask(case["query"], k=3)
            expected = case["expected_answer"].lower()
            actual = answer.lower()
            
            # Simple keyword overlap score
            expected_words = set(expected.split())
            actual_words = set(actual.split())
            
            if expected_words:
                overlap = len(expected_words & actual_words)
                score = overlap / len(expected_words)
                total_score += score
                evaluated += 1
        
        return total_score / evaluated if evaluated > 0 else 0.0
    
    def detailed_report(self, test_file: str) -> str:
        """Generate detailed evaluation report."""
        results = self.evaluate(test_file)
        
        report = f"""
📊 RAG Evaluation Report
{'='*30}

📈 Retrieval Metrics:
  Recall@1: {results.recall_at_1:.1%}
  Recall@3: {results.recall_at_3:.1%} 
  Recall@5: {results.recall_at_5:.1%}

💬 Answer Quality:
  Relevance Score: {results.answer_relevance:.1%}
  
📋 Summary:
  Total Queries: {results.total_queries}
  
💡 Insights:
  - Retrieval is {'excellent' if results.recall_at_3 >= 0.9 else 'good' if results.recall_at_3 >= 0.7 else 'needs improvement'}
  - Answer quality is {'good' if results.answer_relevance >= 0.5 else 'needs improvement'}
"""
        return report