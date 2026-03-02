"""PDF Documentation MCP Server.

Provides semantic search over PDF documentation
via the Model Context Protocol (MCP).
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from pdf2mcp.config import get_settings
from pdf2mcp.search import (
    format_results,
    get_document_sections,
    list_ingested_documents,
    search_documents,
)

__all__ = ["mcp", "run_server"]

logger = logging.getLogger(__name__)

mcp = FastMCP(name="pdf-docs")


@mcp.tool()
def search_docs(query: str, num_results: int = 5) -> str:
    """Search ingested PDF documentation.

    Performs semantic search across all ingested PDF documents.
    Returns relevant passages with source file, section, and page information.

    Use this tool when you need to find information in the ingested documents.

    Args:
        query: Natural language search query describing what you're looking for.
        num_results: Number of results to return (default: 5, max: 20).
    """
    num_results = min(max(num_results, 1), 20)

    try:
        settings = get_settings()
        results = search_documents(query, settings, num_results=num_results)
        return format_results(results)
    except Exception:
        logger.exception("Search failed")
        return (
            "Search failed. Make sure documents are ingested and OPENAI_API_KEY is set."
        )


@mcp.tool()
def list_docs() -> str:
    """List all ingested documentation.

    Returns a list of all PDF documents that have been processed and
    are available for search, including chunk counts and file hashes.
    """
    try:
        settings = get_settings()
        docs = list_ingested_documents(settings)
        if not docs:
            return "No documents ingested yet. Run 'pdf2mcp ingest' to process PDFs."

        lines = ["Ingested Documents:", ""]
        for doc in docs:
            lines.append(
                f"- {doc['filename']} "
                f"({doc.get('chunk_count', '?')} chunks, "
                f"hash: {doc.get('file_hash', '?')[:8]}...)"
            )
        lines.append(f"\nTotal: {len(docs)} documents")
        return "\n".join(lines)
    except Exception:
        logger.exception("List documents failed")
        return "Failed to list documents."


@mcp.tool()
def get_sections(filename: str) -> str:
    """Get the section structure of a specific document.

    Returns all section headings found in the specified PDF document,
    in the order they appear. Useful for understanding the document's
    structure before searching for specific content.

    Args:
        filename: The PDF filename (e.g., 'manual.pdf').
    """
    try:
        settings = get_settings()
        sections = get_document_sections(filename, settings)
        if not sections:
            return (
                f"No sections found for '{filename}'. "
                "Check the filename with list_docs."
            )

        lines = [f"Sections in {filename}:", ""]
        for i, section in enumerate(sections, 1):
            lines.append(f"  {i}. {section}")
        return "\n".join(lines)
    except Exception:
        logger.exception("Get sections failed")
        return f"Failed to get sections for '{filename}'."


@mcp.resource("docs://status")
def get_status() -> str:
    """Current status of the documentation server."""
    try:
        settings = get_settings()
        docs = list_ingested_documents(settings)
        total_chunks = sum(doc.get("chunk_count", 0) for doc in docs)
        return (
            f"PDF Docs MCP Server\n"
            f"Documents: {len(docs)}\n"
            f"Total chunks: {total_chunks}\n"
            f"Embedding model: {settings.embedding_model}\n"
            f"Docs directory: {settings.docs_dir}"
        )
    except Exception:
        return "Server running, but no documents ingested yet."


def run_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    name: str | None = None,
) -> None:
    """Start the MCP server.

    Args:
        transport: Transport protocol ("stdio" or "streamable-http").
        host: Host to bind to (only used for HTTP transport).
        port: Port to bind to (only used for HTTP transport).
        name: Override the server name.
    """
    if name:
        mcp._mcp_server.name = name

    if transport == "stdio":
        mcp.run()
    else:
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
