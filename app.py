#!/usr/bin/env python3
import os
import sys
import argparse
import json
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Haystack v2 imports ---
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

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

def load_docs_haystack(path: str) -> List[Document]:
    docs: List[Document] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(id=obj["id"], content=obj["content"]))
    return docs

def build_haystack(docs: List[Document], model_name: str):
    store = InMemoryDocumentStore()
    # Embed documents
    doc_embedder = SentenceTransformersDocumentEmbedder(model=model_name, progress_bar=False)
    doc_embedder.warm_up()
    embedded = doc_embedder.run(documents=docs)
    store.write_documents(embedded["documents"])

    retriever = InMemoryEmbeddingRetriever(document_store=store, top_k=4)
    q_embedder = SentenceTransformersTextEmbedder(model=model_name, progress_bar=False)
    q_embedder.warm_up()
    return store, retriever, q_embedder

def haystack_top_k(retriever: InMemoryEmbeddingRetriever, q_embedder: SentenceTransformersTextEmbedder,
                   query: str, k: int):
    q = q_embedder.run(text=query)
    result = retriever.run(query_embedding=q["embedding"], top_k=k)
    # result["documents"] -> list[Document] with .content and maybe .score
    hits = []
    for d in result["documents"]:
        score = getattr(d, "score", None)
        hits.append(({"id": d.id, "content": d.content}, float(score) if score is not None else None))
    return hits

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
    p = argparse.ArgumentParser(description="Mini RAG (Haystack v2 retrieval + OpenAI generation)")
    p.add_argument("question", type=str, help="Your question")
    p.add_argument("--k", type=int, default=4, help="retriever top_k")
    p.add_argument("--show-sources", action="store_true")
    args = p.parse_args()

    # Load & build Haystack v2 components
    hs_docs = load_docs_haystack(DOCS_PATH)
    _, retriever, q_embedder = build_haystack(hs_docs, EMBED_MODEL)

    # Retrieve with Haystack
    hits = haystack_top_k(retriever, q_embedder, args.question, args.k)
    top_docs = [d for (d, _) in hits]

    # Display question
    print(f"\n{Colors.BOLD}{Colors.CYAN}Question:{Colors.RESET} {args.question}")
    
    # Display retrieval results
    print(f"\n{Colors.BOLD}{Colors.BLUE}Top {args.k} Retrieved Documents:{Colors.RESET}")
    print("─" * 80)
    for i, (d, score) in enumerate(hits, 1):
        short = d["content"]
        if len(short) > 100:
            short = short[:97] + "..."
        if score is not None:
            score_color = Colors.GREEN if score > 0.3 else Colors.YELLOW if score > 0.2 else Colors.DIM
            score_txt = f"{score_color}(score: {score:.3f}){Colors.RESET}"
        else:
            score_txt = f"{Colors.DIM}(score: N/A){Colors.RESET}"
        print(f"  {Colors.BOLD}{i}.{Colors.RESET} [{Colors.CYAN}id={d['id']}{Colors.RESET}] {score_txt}")
        print(f"     {Colors.DIM}{short}{Colors.RESET}")
    
    # Generate and display answer
    print(f"\n{Colors.BOLD}{Colors.GREEN}Generating answer...{Colors.RESET}")
    answer = generate_answer(args.question, top_docs)
    print(f"\n{Colors.BOLD}{Colors.GREEN}Answer:{Colors.RESET}")
    print("─" * 80)
    print(f"{answer}")
    print("─" * 80)

    if args.show_sources:
        print(f"\n{Colors.BOLD}{Colors.HEADER}Source Documents:{Colors.RESET}")
        print("─" * 80)
        for i, d in enumerate(top_docs, 1):
            print(f"  {Colors.BOLD}[{Colors.CYAN}{d['id']}{Colors.RESET}{Colors.BOLD}]{Colors.RESET} {d['content']}")
            if i < len(top_docs):
                print()

if __name__ == "__main__":
    main()