#!/usr/bin/env python3
import json
from typing import List, Dict, Tuple
from app import load_docs, DenseRetriever, EMBED_MODEL, DOCS_PATH

GOLDEN_PATH = "data/golden_test.json"

def load_golden(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["test_cases"]

def recall_at_k(retriever: DenseRetriever, golden: List[Dict[str, object]], k: int) -> float:
    hits = 0
    for case in golden:
        q = str(case["query"])
        relevant = set(case["relevant_doc_ids"])
        top = retriever.top_k(q, k=k)
        returned_ids = {d["id"] for (d, _) in top}
        if relevant & returned_ids:
            hits += 1
    return hits / len(golden)

if __name__ == "__main__":
    docs = load_docs(DOCS_PATH)
    retr = DenseRetriever(docs, EMBED_MODEL)
    golden = load_golden(GOLDEN_PATH)
    
    print("Evaluation Results:")
    print("-" * 20)
    for k in (1, 3, 5):
        r = recall_at_k(retr, golden, k)
        print(f"Recall@{k}: {r:.2f}")
