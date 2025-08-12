"""Command line interface."""

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
@click.option("--refresh-cache", is_flag=True, help="Refresh embedding cache")
@click.option(
    "--source",
    default="data/docs.jsonl",
    help="Path to documents (file, directory, or URL)",
)
@click.option(
    "--source-type",
    default="auto",
    help="Source type: auto, jsonl, pdf, docx, txt, url, directory",
)
def ask(
    question: str,
    k: int,
    show_sources: bool,
    refresh_cache: bool,
    source: str,
    source_type: str,
):
    """Ask a question."""
    click.echo("🔄 Loading...")

    rag = SimpleRAG(docs_path=source, source_type=source_type)
    rag.setup(force_refresh=refresh_cache)

    click.echo("🔍 Searching...")
    answer, sources = rag.ask(question, k)

    # Display results
    click.echo(f"\n❓ Question: {question}")
    click.echo(f"\n📚 Top {k} results:")
    for i, source in enumerate(sources, 1):
        score = f"({source.score:.3f})" if source.score else ""
        preview = (
            source.content[:100] + "..."
            if len(source.content) > 100
            else source.content
        )
        click.echo(f"  {i}. [{source.id}] {score} {preview}")

    click.echo(f"\n✨ Answer:\n{answer}")

    if show_sources:
        click.echo("\n📖 Full Sources:")
        for source in sources:
            click.echo(f"  [{source.id}] {source.content}")


@main.command()
@click.option("--detailed", is_flag=True, help="Show detailed evaluation report")
@click.option("--refresh-cache", is_flag=True, help="Refresh embedding cache")
@click.option(
    "--source",
    default="data/docs.jsonl",
    help="Path to documents (file, directory, or URL)",
)
@click.option(
    "--source-type",
    default="auto",
    help="Source type: auto, jsonl, pdf, docx, txt, url, directory",
)
def eval(detailed: bool, refresh_cache: bool, source: str, source_type: str):
    """Evaluate the RAG system."""
    from .evaluator import Evaluator

    click.echo("📊 Running evaluation...")

    rag = SimpleRAG(docs_path=source, source_type=source_type)
    rag.setup(force_refresh=refresh_cache)
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


@main.command()
@click.option("--clear", is_flag=True, help="Clear all cached embeddings")
def cache(clear: bool):
    """Manage embedding cache."""
    from .cache import EmbeddingCache

    cache_manager = EmbeddingCache()

    if clear:
        cache_manager.clear_cache()
        click.echo("🗑️  Cleared all cached embeddings")
        return

    # Show cache info
    info = cache_manager.cache_info()
    click.echo("📦 Cache Info:")
    click.echo(f"  Location: {info['cache_dir']}")
    click.echo(f"  Files: {info['num_files']}")
    click.echo(f"  Size: {info['total_size_mb']:.1f} MB")


if __name__ == "__main__":
    main()
