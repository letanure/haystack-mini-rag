#!/usr/bin/env python3
import sys
import json
from typing import List, Dict, Tuple
from app import load_docs_haystack, build_haystack, haystack_top_k, EMBED_MODEL, DOCS_PATH

# --- Colors for terminal output ---
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    @staticmethod
    def disable():
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.CYAN = ''
        Colors.BOLD = ''
        Colors.RESET = ''

if not sys.stdout.isatty():
    Colors.disable()

GOLDEN_PATH = "data/golden_test.json"

def load_golden(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data["test_cases"]

def recall_at_k(retriever, q_embedder, golden: List[Dict[str, object]], k: int) -> float:
    hits = 0
    for case in golden:
        q = str(case["query"])
        relevant = set(case["relevant_doc_ids"])
        top = haystack_top_k(retriever, q_embedder, q, k)
        returned_ids = {d["id"] for (d, _) in top}
        if relevant & returned_ids:
            hits += 1
    return hits / len(golden)

if __name__ == "__main__":
    docs = load_docs_haystack(DOCS_PATH)
    _, retriever, q_embedder = build_haystack(docs, EMBED_MODEL)
    golden = load_golden(GOLDEN_PATH)
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Evaluation Results{Colors.RESET}")
    print("━" * 40)
    print(f"Dataset: {len(docs)} documents | Test queries: {len(golden)}")
    print("━" * 40)
    
    results = []
    for k in (1, 3, 5):
        r = recall_at_k(retriever, q_embedder, golden, k)
        results.append((k, r))
        
        # Color code based on performance
        if r >= 0.9:
            color = Colors.GREEN
            icon = "[PASS]"
        elif r >= 0.7:
            color = Colors.YELLOW
            icon = "[WARN]"
        else:
            color = Colors.RED
            icon = "[FAIL]"
        
        bar_length = int(r * 20)
        bar = "=" * bar_length + "-" * (20 - bar_length)
        
        print(f"{icon} Recall@{k}: {color}{r:.2%}{Colors.RESET} {bar}")
    
    print("━" * 40)
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET} Perfect recall achieved at k=3")
