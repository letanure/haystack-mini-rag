#!/usr/bin/env python3
import os
import argparse
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer

DOCS_PATH = "data/docs.jsonl"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_docs(path: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append({"id": obj["id"], "content": obj["content"]})
    return docs

def normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

class DenseRetriever:
    def __init__(self, docs: List[Dict[str, str]], model_name: str = EMBED_MODEL) -> None:
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        texts = [d["content"] for d in docs]
        emb = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        self.doc_vecs = normalize(np.array(emb, dtype=np.float32))
        self.ids = [d["id"] for d in docs]

    def top_k(self, query: str, k: int = 4) -> List[Tuple[Dict[str, str], float]]:
        qv = self.model.encode([query], show_progress_bar=False)
        qv = normalize(np.array(qv, dtype=np.float32))[0]
        sims = (self.doc_vecs @ qv)
        idx = np.argsort(-sims)[:k]
        return [ (self.docs[i], float(sims[i])) for i in idx ]

def main():
    load_dotenv()
    p = argparse.ArgumentParser(description="Mini RAG (retrieval-only)")
    p.add_argument("question", type=str, help="Your question")
    p.add_argument("--k", type=int, default=4, help="retriever top_k")
    p.add_argument("--show-sources", action="store_true")
    args = p.parse_args()

    docs = load_docs(DOCS_PATH)
    retriever = DenseRetriever(docs, EMBED_MODEL)
    hits = retriever.top_k(args.question, k=args.k)

    print(f"Q: {args.question}")
    print("Top matches:")
    for i, (d, score) in enumerate(hits, 1):
        short = d["content"]
        if len(short) > 120:
            short = short[:117] + "..."
        print(f"{i}. [id={d['id']}] cos={score:.3f} — {short}")

    if args.show_sources:
        print("\n-- SOURCES --")
        for d, _ in hits:
            print(f"[{d['id']}] {d['content']}")

if __name__ == "__main__":
    main()