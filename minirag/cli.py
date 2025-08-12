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
@click.option("--detailed", is_flag=True, help="Show detailed evaluation report")
def eval(detailed: bool):
    """Evaluate the RAG system."""
    from .evaluator import Evaluator
    
    click.echo("📊 Running evaluation...")
    
    rag = SimpleRAG()
    rag.setup()
    evaluator = Evaluator(rag)
    
    if detailed:
        report = evaluator.detailed_report("data/golden_test.json")
        click.echo(report)
    else:
        results = evaluator.evaluate("data/golden_test.json")
        click.echo("-" * 40)
        click.echo(f"Recall@1:  {results.recall_at_1:.1%}")
        click.echo(f"Recall@3:  {results.recall_at_3:.1%}")  
        click.echo(f"Recall@5:  {results.recall_at_5:.1%}")
        click.echo(f"Answer Quality: {results.answer_relevance:.1%}")
        click.echo(f"Total Queries:  {results.total_queries}")


if __name__ == "__main__":
    main()