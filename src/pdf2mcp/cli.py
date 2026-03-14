"""Command-line interface for pdf2mcp."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pdf2mcp import __version__

if TYPE_CHECKING:
    from pdf2mcp.config import ServerSettings

_ENV_TEMPLATE = """\
# ── Server settings (used by the pdf2mcp server process) ──────────────

# Required: OpenAI API key for embeddings
OPENAI_API_KEY=sk-your-api-key-here

# Optional: OpenAI base URL (for Azure, local proxies, or compatible providers)
# PDF2MCP_OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Override server defaults
# PDF2MCP_DOCS_DIR=docs
# PDF2MCP_DATA_DIR=data
# PDF2MCP_EMBEDDING_MODEL=text-embedding-3-small
# PDF2MCP_CHUNK_SIZE=500
# PDF2MCP_CHUNK_OVERLAP=50
# PDF2MCP_DEFAULT_NUM_RESULTS=5
# PDF2MCP_SERVER_NAME=pdf-docs
# PDF2MCP_SERVER_HOST=127.0.0.1
# PDF2MCP_SERVER_PORT=8000

# Optional: Search mode (semantic, hybrid, or keyword)
# PDF2MCP_SEARCH_MODE=semantic

# Optional: OCR settings (for scanned/image-only PDFs — requires Tesseract)
# PDF2MCP_OCR_ENABLED=true
# PDF2MCP_OCR_LANGUAGE=eng
# PDF2MCP_OCR_DPI=300
"""


def setup_logging(verbose: bool = False) -> None:
    """Configure logging to stderr (stdout reserved for MCP stdio transport)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _load_settings() -> ServerSettings:
    """Load settings or exit with an error message."""
    logger = logging.getLogger("pdf2mcp")
    try:
        from pdf2mcp.config import get_settings

        return get_settings()
    except Exception as exc:  # noqa: BLE001
        logger.error("Configuration error: %s", exc)
        logger.error(
            "Make sure OPENAI_API_KEY is set (in .env or environment variable)"
        )
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline."""
    setup_logging(args.verbose)
    logger = logging.getLogger("pdf2mcp")

    settings = _load_settings()

    # Override docs_dir if --docs-dir was provided
    if args.docs_dir is not None:
        settings.docs_dir = Path(args.docs_dir)

    docs_dir = settings.docs_dir
    if not docs_dir.exists():
        logger.error("Docs directory not found: %s", docs_dir)
        logger.error("Create the directory and add your PDF files.")
        sys.exit(1)

    logger.info("Docs directory: %s", settings.docs_dir)
    logger.info("Data directory: %s", settings.data_dir)

    from pdf2mcp.ingest import run_ingestion

    try:
        run_ingestion(settings, force=args.force, show_progress=True)
        logger.info("Ingestion completed successfully")
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    setup_logging(args.verbose)
    logger = logging.getLogger("pdf2mcp")

    settings = _load_settings()

    # Override docs_dir if --docs-dir was provided
    if args.docs_dir is not None:
        settings.docs_dir = Path(args.docs_dir)

    transport = args.transport or settings.server_transport
    host = args.host or settings.server_host
    port = args.port or settings.server_port
    name = args.name or settings.server_name

    logger.info("Starting PDF Docs MCP server (transport=%s)...", transport)

    from pdf2mcp.server import run_server

    run_server(transport=transport, host=host, port=port, name=name)


def cmd_config(args: argparse.Namespace) -> None:
    """Print MCP client configuration snippets.

    Defaults to HTTP (streamable-http) — the recommended transport when
    server and client are separate processes.  Use ``--transport stdio``
    for the legacy all-in-one mode where the client spawns the server.
    """
    from pdf2mcp.interactive import (
        CLIENT_CHOICES,
        CLIENT_FILES,
        build_config_snippet,
    )

    logger = logging.getLogger("pdf2mcp")

    try:
        from pdf2mcp.config import get_settings

        settings = get_settings()
    except Exception as exc:  # noqa: BLE001
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    name = args.name or settings.server_name
    transport = args.transport or settings.server_transport
    url = args.url or f"http://{settings.server_host}:{settings.server_port}/mcp"

    client_labels = dict(CLIENT_CHOICES)
    clients = (
        [args.client]
        if args.client
        else ["claude-code", "claude-desktop", "cursor", "vscode"]
    )

    for client in clients:
        snippet = build_config_snippet(client, name, transport, url)
        label = client_labels.get(client, client)
        file_hint = CLIENT_FILES.get(client, "config.json")
        print(f"\n# {label} ({file_hint})")
        print(json.dumps(snippet, indent=2))

        if client == "claude-desktop" and transport != "stdio":
            print(
                "# Note: Claude Desktop requires stdio. "
                "Start the server separately and use another client, "
                "or pass --transport stdio."
            )


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete a document from the index."""
    setup_logging(args.verbose)
    logger = logging.getLogger("pdf2mcp")

    settings = _load_settings()

    from pdf2mcp.store import (
        delete_by_source,
        delete_ingestion_metadata,
        get_db,
        get_ingested_files,
        invalidate_table_cache,
    )

    db = get_db(settings)
    ingested = get_ingested_files(db)

    filename = args.filename
    if filename not in ingested:
        logger.error("File '%s' not found in index", filename)
        sys.exit(1)

    if not args.yes:
        answer = input(f"Delete '{filename}' from index? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Cancelled.", file=sys.stderr)
            return

    delete_by_source(db, filename)
    delete_ingestion_metadata(db, filename)
    invalidate_table_cache()
    logger.info("Deleted '%s' from index", filename)


def _format_bytes(size: int) -> str:
    """Format a byte count into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f} TB"


def cmd_stats(args: argparse.Namespace) -> None:
    """Display index statistics."""
    setup_logging(args.verbose)

    settings = _load_settings()

    from rich.console import Console
    from rich.table import Table

    from pdf2mcp.search import list_ingested_documents
    from pdf2mcp.store import DOCUMENTS_TABLE, get_db, table_exists

    console = Console(stderr=True)
    db = get_db(settings)
    docs = list_ingested_documents(settings)
    total_chunks = sum(doc.get("chunk_count", 0) for doc in docs)

    # Compute average chunk size
    avg_chunk_size = 0
    if table_exists(db, DOCUMENTS_TABLE) and total_chunks > 0:
        table = db.open_table(DOCUMENTS_TABLE)
        arrow_table = table.to_arrow()
        texts = arrow_table.column("text").to_pylist()
        avg_chunk_size = sum(len(t) for t in texts) // len(texts) if texts else 0

    # Compute DB size on disk
    db_path = settings.data_dir / "lancedb"
    db_size = 0
    if db_path.exists():
        for f in db_path.rglob("*"):
            if f.is_file():
                db_size += f.stat().st_size

    # Summary table
    summary = Table(title="pdf2mcp Index Statistics", show_header=False)
    summary.add_column("Key", style="bold")
    summary.add_column("Value")
    summary.add_row("Documents", str(len(docs)))
    summary.add_row("Total chunks", str(total_chunks))
    summary.add_row("Avg chunk size", f"{avg_chunk_size} chars")
    summary.add_row("Embedding model", settings.embedding_model)
    summary.add_row("Search mode", getattr(settings, "search_mode", "semantic"))
    summary.add_row("Database size", _format_bytes(db_size))
    summary.add_row("Docs directory", str(settings.docs_dir))
    summary.add_row("Data directory", str(settings.data_dir))
    console.print(summary)

    # Per-document table
    if docs:
        doc_table = Table(title="Ingested Documents")
        doc_table.add_column("Filename")
        doc_table.add_column("Chunks", justify="right")
        doc_table.add_column("Hash")
        for doc in docs:
            doc_table.add_row(
                doc["filename"],
                str(doc["chunk_count"]),
                doc.get("file_hash", "?")[:12],
            )
        console.print(doc_table)


def cmd_search(args: argparse.Namespace) -> None:
    """Search the index from the command line."""
    setup_logging(args.verbose)

    settings = _load_settings()

    from pdf2mcp.search import (
        format_results,
        search_documents,
        search_in_document,
    )

    if args.filename:
        results = search_in_document(
            args.query, args.filename, settings, num_results=args.num_results
        )
    else:
        results = search_documents(args.query, settings, num_results=args.num_results)

    print(format_results(results), file=sys.stderr)


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a working directory for pdf2mcp."""
    if getattr(args, "interactive", False):
        _cmd_init_interactive(args)
    else:
        _cmd_init_scaffold(args)


def _cmd_init_scaffold(args: argparse.Namespace) -> None:
    """Non-interactive scaffolding (original behaviour)."""
    target = Path(args.directory)

    docs_dir = target / "docs"
    env_file = target / ".env"

    docs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created {docs_dir}/       \u2190 put your PDFs here", file=sys.stderr)

    if not env_file.exists():
        env_file.write_text(_ENV_TEMPLATE)
        print(
            f"Created {env_file}        \u2190 add your OPENAI_API_KEY",
            file=sys.stderr,
        )
    else:
        print(f"Skipped {env_file}   (already exists)", file=sys.stderr)

    print(
        "\nNext steps:\n"
        f"  1. Add your PDFs to {docs_dir}/\n"
        f"  2. Set OPENAI_API_KEY in {env_file}\n"
        f"  3. cd {target} && pdf2mcp ingest\n"
        "  4. pdf2mcp config",
        file=sys.stderr,
    )


def _cmd_init_interactive(args: argparse.Namespace) -> None:
    """Interactive setup wizard."""
    from pdf2mcp.interactive import (
        WizardCancelledError,
        apply_wizard_result,
        run_post_setup,
        run_wizard,
    )

    try:
        result = run_wizard(Path(args.directory))
        apply_wizard_result(result)
        run_post_setup(result)
    except WizardCancelledError:
        print("\nSetup cancelled.", file=sys.stderr)
        sys.exit(130)


_BANNER = r"""
██████╗ ██████╗ ███████╗██████╗ ███╗   ███╗ ██████╗██████╗
██╔══██╗██╔══██╗██╔════╝╚════██╗████╗ ████║██╔════╝██╔══██╗
██████╔╝██║  ██║█████╗   █████╔╝██╔████╔██║██║     ██████╔╝
██╔═══╝ ██║  ██║██╔══╝  ██╔═══╝ ██║╚██╔╝██║██║     ██╔═══╝
██║     ██████╔╝██║     ███████╗██║ ╚═╝ ██║╚██████╗██║
╚═╝     ╚═════╝ ╚═╝     ╚══════╝╚═╝     ╚═╝ ╚═════╝╚═╝
"""


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pdf2mcp",
        description=_BANNER + "Turn any PDF folder into a searchable MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest", help="Process PDFs and build the search index"
    )
    ingest_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Clear database and re-ingest all documents",
    )
    ingest_parser.add_argument(
        "--docs-dir",
        default=None,
        help="Override docs directory (default: from config or ./docs)",
    )

    # serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start the MCP server")
    serve_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=None,
        help="Transport protocol (default: from config or stdio)",
    )
    serve_parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to for HTTP transport (default: from config or 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to for HTTP transport (default: from config or 8000)",
    )
    serve_parser.add_argument(
        "--name",
        default=None,
        help="MCP server name (default: from config or pdf-docs)",
    )
    serve_parser.add_argument(
        "--docs-dir",
        default=None,
        help="Override docs directory (default: from config or ./docs)",
    )

    # config subcommand
    config_parser = subparsers.add_parser(
        "config", help="Print MCP client configuration snippets"
    )
    config_parser.add_argument(
        "--name",
        default=None,
        help="Server name for config snippets (default: pdf-docs)",
    )
    config_parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=None,
        help="Transport protocol (default: streamable-http)",
    )
    config_parser.add_argument(
        "--url",
        default=None,
        help="Server URL for HTTP transport (default: http://127.0.0.1:8000/mcp)",
    )
    config_parser.add_argument(
        "--client",
        choices=["claude-code", "claude-desktop", "cursor", "vscode"],
        default=None,
        help="Print config for a specific client only",
    )

    # delete subcommand
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a document from the index"
    )
    delete_parser.add_argument("filename", help="PDF filename to delete")
    delete_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    delete_parser.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )

    # stats subcommand
    stats_parser = subparsers.add_parser("stats", help="Display index statistics")
    stats_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    # search subcommand
    search_parser = subparsers.add_parser(
        "search", help="Search the index from the command line"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-n",
        "--num-results",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    search_parser.add_argument(
        "--filename",
        default=None,
        help="Restrict search to a specific document",
    )
    search_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    # init subcommand
    init_parser = subparsers.add_parser(
        "init", help="Scaffold a working directory for pdf2mcp"
    )
    init_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )
    init_parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch guided setup wizard",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
