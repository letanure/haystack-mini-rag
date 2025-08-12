#!/usr/bin/env python3
import os
import sys
import argparse
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sentence_transformers import SentenceTransformer

# --- Colors for terminal output ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    @staticmethod
    def disable():
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.BOLD = ''
        Colors.DIM = ''
        Colors.RESET = ''

# Disable colors if output is not a terminal
if not sys.stdout.isatty():
    Colors.disable()

# --- Config ---
DOCS_PATH = "data/docs.jsonl"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- OpenAI (chat) ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # handled below

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
        return [(self.docs[i], float(sims[i])) for i in idx]

def build_prompt(question: str, sources: List[Dict[str, str]]) -> str:
    joined = "\n".join([f"[{d['id']}] {d['content']}" for d in sources])
    return (
        "You are a precise assistant. Use ONLY the provided context.\n"
        "If the answer is not in the context, say you don't know.\n"
        "Cite sources like [id] when possible.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{joined}\n\n"
        "Answer:"
    )


def generate_answer(question: str, sources: List[Dict[str, str]]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"{Colors.RED}[error]{Colors.RESET} OPENAI_API_KEY not set. Create .env and export the key."

    if OpenAI is None:
        return f"{Colors.RED}[error]{Colors.RESET} openai package is not installed. Run: pip install openai"

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(question, sources)

    # Chat Completions API (widely supported)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful, concise assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


def main():
    load_dotenv()
    p = argparse.ArgumentParser(description="Mini RAG (retrieval + generation)")
    p.add_argument("question", type=str, help="Your question")
    p.add_argument("--k", type=int, default=4, help="retriever top_k")
    p.add_argument("--show-sources", action="store_true")
    args = p.parse_args()

    docs = load_docs(DOCS_PATH)
    retriever = DenseRetriever(docs, EMBED_MODEL)
    hits = retriever.top_k(args.question, k=args.k)
    top_docs = [d for (d, _) in hits]

    # Display question
    print(f"\n{Colors.BOLD}{Colors.CYAN}❓ Question:{Colors.RESET} {args.question}")
    
    # Display retrieval results
    print(f"\n{Colors.BOLD}{Colors.BLUE}🔍 Top {args.k} Retrieved Documents:{Colors.RESET}")
    print("─" * 80)
    for i, (d, score) in enumerate(hits, 1):
        short = d["content"]
        if len(short) > 100:
            short = short[:97] + "..."
        score_color = Colors.GREEN if score > 0.3 else Colors.YELLOW if score > 0.2 else Colors.DIM
        print(f"  {Colors.BOLD}{i}.{Colors.RESET} [{Colors.CYAN}id={d['id']}{Colors.RESET}] {score_color}(score: {score:.3f}){Colors.RESET}")
        print(f"     {Colors.DIM}{short}{Colors.RESET}")
    
    # Generate and display answer
    print(f"\n{Colors.BOLD}{Colors.GREEN}💡 Generating answer...{Colors.RESET}")
    answer = generate_answer(args.question, top_docs)
    print(f"\n{Colors.BOLD}{Colors.GREEN}✨ Answer:{Colors.RESET}")
    print("─" * 80)
    print(f"{answer}")
    print("─" * 80)

    if args.show_sources:
        print(f"\n{Colors.BOLD}{Colors.HEADER}📚 Source Documents:{Colors.RESET}")
        print("─" * 80)
        for i, d in enumerate(top_docs, 1):
            print(f"  {Colors.BOLD}[{Colors.CYAN}{d['id']}{Colors.RESET}{Colors.BOLD}]{Colors.RESET} {d['content']}")
            if i < len(top_docs):
                print()

if __name__ == "__main__":
    main()