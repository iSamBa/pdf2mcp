"""Command-line interface for pdf2mcp."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pdf2mcp import __version__

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

# ── Client settings (used by MCP clients to connect) ─────────────────

# PDF2MCP_CLIENT_SERVER_NAME=pdf-docs
# PDF2MCP_CLIENT_SERVER_URL=http://127.0.0.1:8000/mcp
# PDF2MCP_CLIENT_TRANSPORT=streamable-http
"""


def setup_logging(verbose: bool = False) -> None:
    """Configure logging to stderr (stdout reserved for MCP stdio transport)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def cmd_ingest(args: argparse.Namespace) -> None:
    """Run the ingestion pipeline."""
    setup_logging(args.verbose)
    logger = logging.getLogger("pdf2mcp")

    try:
        from pdf2mcp.config import get_settings

        settings = get_settings()
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        logger.error(
            "Make sure OPENAI_API_KEY is set (in .env or environment variable)"
        )
        sys.exit(1)

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
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    setup_logging(args.verbose)
    logger = logging.getLogger("pdf2mcp")

    try:
        from pdf2mcp.config import get_settings

        settings = get_settings()
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

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
    from pdf2mcp.config import ClientSettings

    # Start from ClientSettings defaults, then apply CLI overrides
    cs = ClientSettings()
    name = args.name or cs.server_name
    transport = args.transport or cs.transport
    url = args.url or cs.server_url

    clients = (
        [args.client]
        if args.client
        else ["claude-code", "claude-desktop", "cursor", "vscode"]
    )

    for client in clients:
        snippet = _build_config_snippet(client, name, transport, url)
        label = _client_label(client)
        file_hint = _client_file(client)
        print(f"\n# {label} ({file_hint})")
        print(json.dumps(snippet, indent=2))

        if client == "claude-desktop" and transport != "stdio":
            print(
                "# Note: Claude Desktop requires stdio. "
                "Start the server separately and use another client, "
                "or pass --transport stdio."
            )


def _build_config_snippet(
    client: str,
    name: str,
    transport: str,
    url: str,
) -> dict[str, object]:
    """Build a config snippet for a specific client."""
    is_http = transport != "stdio"

    if client == "vscode":
        top_key = "servers"
    else:
        top_key = "mcpServers"

    if is_http:
        server_config: dict[str, object] = {
            "type": "http",
            "url": url,
        }
    else:
        server_config = {
            "command": "uv",
            "args": ["run", "pdf2mcp", "serve"],
        }

    return {top_key: {name: server_config}}


def _client_label(client: str) -> str:
    labels = {
        "claude-code": "Claude Code",
        "claude-desktop": "Claude Desktop",
        "cursor": "Cursor",
        "vscode": "VS Code / GitHub Copilot",
    }
    return labels.get(client, client)


def _client_file(client: str) -> str:
    files = {
        "claude-code": ".mcp.json",
        "claude-desktop": "claude_desktop_config.json",
        "cursor": ".cursor/mcp.json",
        "vscode": ".vscode/mcp.json",
    }
    return files.get(client, "config.json")


def cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a working directory for pdf2mcp."""
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

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "init":
        cmd_init(args)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
