"""Command line interface."""

import json
import click
from dotenv import load_dotenv

from .rag import SimpleRAG


@click.group()
def main():
    """Simple RAG system."""
    load_dotenv()


@main.command()
@click.argument("question")
@click.option("--k", default=4, help="Number of docs to retrieve")
@click.option("--show-sources", is_flag=True, help="Show source documents")
def ask(question: str, k: int, show_sources: bool):
    """Ask a question."""
    click.echo("🔄 Loading...")
    
    rag = SimpleRAG()
    rag.setup()
    
    click.echo("🔍 Searching...")
    answer, sources = rag.ask(question, k)
    
    # Display results
    click.echo(f"\n❓ Question: {question}")
    click.echo(f"\n📚 Top {k} results:")
    for i, source in enumerate(sources, 1):
        score = f"({source.score:.3f})" if source.score else ""
        preview = source.content[:100] + "..." if len(source.content) > 100 else source.content
        click.echo(f"  {i}. [{source.id}] {score} {preview}")
    
    click.echo(f"\n✨ Answer:\n{answer}")
    
    if show_sources:
        click.echo(f"\n📖 Full Sources:")
        for source in sources:
            click.echo(f"  [{source.id}] {source.content}")


@main.command()
def eval():
    """Evaluate the RAG system."""
    click.echo("📊 Running evaluation...")
    
    rag = SimpleRAG()
    rag.setup()
    
    # Load test data
    with open("data/golden_test.json") as f:
        test_data = json.load(f)["test_cases"]
    
    click.echo("-" * 30)
    
    for k in [1, 3, 5]:
        hits = 0
        for case in test_data:
            sources = rag.search(case["query"], k)
            returned_ids = {s.id for s in sources}
            relevant_ids = set(case["relevant_doc_ids"])
            if returned_ids & relevant_ids:
                hits += 1
        
        recall = hits / len(test_data)
        click.echo(f"Recall@{k}: {recall:.1%} ({hits}/{len(test_data)})")


if __name__ == "__main__":
    main()