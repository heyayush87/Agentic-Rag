"""RetailIQ command-line interface.

    retailiq ingest --reset
    retailiq ask "How long do I have to return an electrical item?"
    retailiq chat
    retailiq evaluate
    retailiq config
    retailiq serve

Heavy imports are deferred into each command body so `retailiq --help`
responds instantly instead of loading LangChain, Chroma and torch first.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from retailiq import __version__

app = typer.Typer(
    name="retailiq",
    help="RetailIQ — agentic RAG assistant for retail operations.",
    add_completion=False,
)
console = Console()
error_console = Console(stderr=True)


def _bootstrap() -> None:
    """Configure logging from settings before any command runs."""
    from retailiq.core.logging import configure_logging
    from retailiq.core.settings import get_settings

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)


def _fail(exc: Exception) -> None:
    """Print a platform error readably and exit non-zero."""
    from retailiq.core.exceptions import RetailIQError

    if isinstance(exc, RetailIQError):
        error_console.print(f"[bold red]{exc.code}[/bold red]: {exc.message}")
    else:
        error_console.print(f"[bold red]error[/bold red]: {exc}")
    raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show the version and exit.")] = False,
) -> None:
    """Root callback.

    `invoke_without_command` lets `--version` run on its own; without it
    Typer insists on a subcommand and rejects the bare flag.
    """
    if version:
        console.print(f"RetailIQ {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def ingest(
    reset: Annotated[
        bool,
        typer.Option(
            "--reset/--append",
            help="Drop the existing index first. Appending duplicates every chunk.",
        ),
    ] = True,
) -> None:
    """Build the vector index from the knowledge base."""
    _bootstrap()
    from retailiq.ingestion.pipeline import build_index

    try:
        with console.status("Ingesting documents…"):
            result = build_index(reset=reset)
    except Exception as exc:
        _fail(exc)
    else:
        console.print(f"[green]✓[/green] {result.describe()}")


@app.command()
def ask(
    question: Annotated[list[str], typer.Argument(help="The question to ask.")],
    trace: Annotated[
        bool, typer.Option("--trace/--no-trace", help="Show the decision trace.")
    ] = True,
    context: Annotated[bool, typer.Option("--context", help="Show the retrieved chunks.")] = False,
) -> None:
    """Ask a single question."""
    _bootstrap()
    from retailiq.services.rag_service import RAGService

    text = " ".join(question).strip()
    try:
        with console.status("The agent is reasoning…"):
            result = RAGService().answer(text)
    except Exception as exc:
        _fail(exc)
        return

    console.print(Panel(result.answer, title=f"[bold]{text}[/bold]", border_style="green"))

    if result.sources:
        console.print(f"[dim]sources:[/dim] {', '.join(result.sources)}")
    console.print(
        f"[dim]route={result.route}  grounded={result.grounded}  "
        f"rewrites={result.rewrites}  {result.latency_ms}ms[/dim]"
    )

    if trace and result.trace:
        table = Table(title="Agent decision trace", show_header=False, border_style="dim")
        for index, step in enumerate(result.trace, start=1):
            table.add_row(str(index), step.detail)
        console.print(table)

    if context:
        console.print(Panel(result.context, title="Retrieved context", border_style="dim"))


@app.command()
def chat() -> None:
    """Start an interactive session."""
    _bootstrap()
    from retailiq.services.rag_service import RAGService

    service = RAGService()
    console.print("[bold]RetailIQ[/bold] — interactive mode. Type 'exit' to quit.\n")

    while True:
        try:
            question = console.input("[cyan]you ›[/cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if question.lower() in {"exit", "quit", ":q"}:
            break
        if not question:
            continue

        try:
            result = service.answer(question)
        except Exception as exc:
            error_console.print(f"[red]error:[/red] {exc}")
            continue

        console.print(f"\n[green]assistant ›[/green] {result.answer}")
        console.print(f"[dim]  [{' | '.join(s.detail for s in result.trace)}][/dim]\n")


@app.command()
def evaluate(
    output: Annotated[
        str | None, typer.Option("--output", "-o", help="Write the JSON report to this path.")
    ] = None,
) -> None:
    """Score the assistant against the golden question set."""
    _bootstrap()
    from retailiq.services.evaluation_service import EvaluationService

    try:
        with console.status("Running evaluation…"):
            report = EvaluationService().run()
    except Exception as exc:
        _fail(exc)
        return

    table = Table(title="Per-question results")
    table.add_column("question", overflow="fold", max_width=46)
    table.add_column("src", justify="center")
    table.add_column("kw", justify="center")
    table.add_column("faith", justify="center")
    table.add_column("rel", justify="center")

    def tick(value: bool) -> str:
        return "[green]✓[/green]" if value else "[red]✗[/red]"

    for row in report.rows:
        table.add_row(
            row.question,
            tick(row.retrieved_expected_source),
            tick(row.keywords_present),
            tick(row.faithful),
            tick(row.relevant),
        )
    console.print(table)

    summary = report.summary()
    console.print("\n[bold]Aggregate metrics[/bold]")
    for metric in ("retrieval_hit_rate", "keyword_recall", "faithfulness", "answer_relevance"):
        console.print(f"  {metric:22} {float(summary[metric]):.0%}")

    if output:
        from pathlib import Path

        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"\n[green]✓[/green] Report written to {path}")


@app.command()
def config() -> None:
    """Show the effective configuration. Secrets are never printed."""
    _bootstrap()
    from retailiq.core.settings import get_settings

    table = Table(title="Effective configuration", show_header=False)
    for key, value in get_settings().describe().items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="Bind address.")] = None,
    port: Annotated[int | None, typer.Option(help="Bind port.")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Auto-reload on code changes.")] = False,
) -> None:
    """Run the REST API server."""
    _bootstrap()
    try:
        import uvicorn
    except ImportError:
        error_console.print(
            "[red]error:[/red] uvicorn is not installed. Install with: pip install 'retailiq[api]'"
        )
        raise typer.Exit(code=1) from None

    from retailiq.core.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        "retailiq.api.main:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        reload=reload,
    )


def run() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    sys.exit(app())
