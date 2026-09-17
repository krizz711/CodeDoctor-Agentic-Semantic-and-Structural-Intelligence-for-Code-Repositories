from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def cmd_index(args):
    """Index a repository."""
    from app.indexing.sqlite_index import SQLiteIndex
    from app.parser.chunker import chunk_source

    repo_path = Path(args.repository)
    if not repo_path.is_dir():
        console.print(f"[red]Error:[/red] '{repo_path}' is not a directory")
        return 1

    index = SQLiteIndex(args.output)
    file_count = 0
    chunk_count = 0

    extensions = {".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".py"}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing files...", total=None)

        for p in repo_path.rglob("*"):
            if any(
                part.startswith(".") or part in ("node_modules", "dist", "build", "coverage", "__pycache__", ".venv")
                for part in p.parts
            ):
                continue
            if p.is_file() and p.suffix in extensions:
                try:
                    source = p.read_text(encoding="utf-8", errors="ignore")
                    rel = str(p.relative_to(repo_path))
                    index.index_file(rel, source)
                    file_count += 1
                    chunks = chunk_source(source, rel)
                    chunk_count += len(chunks)
                    progress.update(task, description=f"Indexed {file_count} files...")
                except Exception as e:
                    console.print(f"  [yellow]Skip:[/yellow] {p} ({e})")

    console.print(
        Panel(
            f"[green]✓[/green] Indexed [bold]{file_count}[/bold] files with [bold]{chunk_count}[/bold] chunks\n"
            f"  Index saved to: [cyan]{args.output}[/cyan]",
            title="Index Complete",
            border_style="green",
        )
    )
    return 0


def cmd_search(args):
    """Search the index."""
    from app.indexing.sqlite_index import SQLiteIndex
    from app.retrieval.hybrid import Document, HybridRetriever

    index = SQLiteIndex(args.index)

    # Get all chunks as documents
    rows = index.conn.execute(
        "SELECT id, name, type, content, start_line, end_line FROM chunks JOIN files ON chunks.file_id = files.id"
    ).fetchall()

    if not rows:
        console.print("[yellow]No chunks in index. Run 'index' first.[/yellow]")
        return 1

    documents = []
    for row in rows:
        file_rows = index.conn.execute(
            "SELECT path FROM files WHERE id = (SELECT file_id FROM chunks WHERE id = ?)", (row["id"],)
        ).fetchall()
        file_path = file_rows[0]["path"] if file_rows else "unknown"
        documents.append(
            Document(
                document_id=row["id"],
                file_path=file_path,
                content=row["content"] or "",
                name=row["name"] or "",
                type=row["type"] or "",
                start_line=row["start_line"] or 1,
                end_line=row["end_line"] or 1,
            )
        )

    retriever = HybridRetriever(documents)
    results = retriever.search(args.query, limit=args.limit)

    if not results:
        console.print(f"[yellow]No results for:[/yellow] {args.query}")
        return 0

    console.print(f"\n[bold]Search results for:[/bold] [cyan]{args.query}[/cyan]\n")

    for i, result in enumerate(results, 1):
        doc = result.document
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        table.add_column(style="dim")
        table.add_column()
        table.add_row("File", f"[cyan]{doc.file_path}[/cyan]")
        table.add_row("Symbol", f"[bold]{doc.name}[/bold] ({doc.type})")
        table.add_row("Lines", f"{doc.start_line}-{doc.end_line}")
        table.add_row("Score", f"{result.score:.4f}")
        table.add_row("Sources", ", ".join(result.sources))

        snippet = doc.content[:500]
        # Detect language for syntax highlighting
        lang = "javascript"
        if doc.file_path.endswith(".py"):
            lang = "python"
        elif doc.file_path.endswith(".ts") or doc.file_path.endswith(".tsx"):
            lang = "typescript"

        syntax = Syntax(snippet, lang, theme="monokai", line_numbers=True, start_line=doc.start_line)

        console.print(
            Panel(
                table,
                title=f"Result {i}",
                subtitle=f"Score: {result.score:.4f}",
                border_style="blue" if i == 1 else "dim",
            )
        )
        console.print(syntax)
        console.print()

    return 0


def cmd_symbol(args):
    """Look up a symbol."""
    from app.indexing.sqlite_index import SQLiteIndex

    index = SQLiteIndex(args.index)
    results = index.search_symbols(args.name)

    if not results:
        console.print(f"[yellow]No symbol found:[/yellow] {args.name}")
        return 1

    table = Table(title=f"Symbols matching '{args.name}'", box=box.ROUNDED)
    table.add_column("Name", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Chunk ID", style="dim")

    for r in results:
        table.add_row(r["name"], r["type"], r.get("chunk_id", ""))

    console.print(table)
    return 0


def cmd_stats(args):
    """Show index statistics."""
    from app.indexing.sqlite_index import SQLiteIndex

    index = SQLiteIndex(args.index)

    files = index.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    chunks = index.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    symbols = index.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    edges = index.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

    table = Table(title="Index Statistics", box=box.ROUNDED)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan", justify="right")
    table.add_row("Files", str(files))
    table.add_row("Chunks", str(chunks))
    table.add_row("Symbols", str(symbols))
    table.add_row("Edges", str(edges))
    table.add_row("Index Path", args.index)

    console.print(table)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="codedoctor",
        description="CodeDoctor - Agentic Code Intelligence",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index
    idx = subparsers.add_parser("index", help="Index a repository")
    idx.add_argument("repository", help="Path to repository")
    idx.add_argument("-o", "--output", default="codedoctor.db", help="Output index path")

    # search
    srch = subparsers.add_parser("search", help="Search the index")
    srch.add_argument("index", help="Path to index file")
    srch.add_argument("query", help="Search query")
    srch.add_argument("-n", "--limit", type=int, default=5, help="Max results")

    # symbol
    sym = subparsers.add_parser("symbol", help="Look up a symbol")
    sym.add_argument("index", help="Path to index file")
    sym.add_argument("name", help="Symbol name")

    # stats
    st = subparsers.add_parser("stats", help="Show index statistics")
    st.add_argument("index", help="Path to index file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    handlers = {
        "index": cmd_index,
        "search": cmd_search,
        "symbol": cmd_symbol,
        "stats": cmd_stats,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
