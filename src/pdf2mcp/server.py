"""PDF Documentation MCP Server.

Provides semantic search over PDF documentation
via the Model Context Protocol (MCP).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from mcp.server.fastmcp import FastMCP

from pdf2mcp.config import get_settings
from pdf2mcp.search import (
    format_page_chunks,
    format_results,
    format_section_chunks,
    get_document_sections,
    get_page_chunks,
    get_section_chunks,
    list_ingested_documents,
    search_documents,
    search_in_document,
)

__all__ = ["mcp", "run_server"]

logger = logging.getLogger(__name__)

mcp = FastMCP(name="pdf-docs")


@mcp.tool()
def search_docs(query: str, num_results: int = 5) -> str:
    """Search across ALL ingested PDFs for relevant passages.

    Use this when you don't know which document contains the answer.
    Returns ranked results with source file, section title, and page numbers.
    Each result includes the matching text passage.

    To search within a specific document instead, use search_in_doc.
    To browse a document's structure first, use get_sections.

    Args:
        query: Natural language search query (e.g., 'safety requirements for
            outdoor installation').
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
def search_in_doc(query: str, filename: str, num_results: int = 5) -> str:
    """Search within a SINGLE document for relevant passages.

    Use this when you already know which PDF to search — avoids noise from
    other documents. Returns ranked results with section title and page numbers.

    Get valid filenames from list_docs first.

    Args:
        query: Natural language search query (e.g., 'maximum torque settings').
        filename: The PDF filename to search within (e.g., 'manual.pdf').
        num_results: Number of results to return (default: 5, max: 20).
    """
    num_results = min(max(num_results, 1), 20)

    try:
        settings = get_settings()
        results = search_in_document(
            query, filename, settings, num_results=num_results
        )
        return format_results(results)
    except Exception:
        logger.exception("Scoped search failed")
        return (
            f"Search in '{filename}' failed. "
            "Check the filename with list_docs and ensure OPENAI_API_KEY is set."
        )


@mcp.tool()
def read_page(filename: str, page: int) -> str:
    """Read the full content of a specific page from a document.

    Use this when you know which page you need — e.g., after seeing a page
    reference in a search result or table of contents. Returns all text chunks
    from that page in document order, grouped by section.

    Get valid filenames from list_docs.

    Args:
        filename: The PDF filename (e.g., 'manual.pdf').
        page: The page number to read (1-indexed).
    """
    try:
        settings = get_settings()
        chunks = get_page_chunks(filename, page, settings)
        return format_page_chunks(chunks, filename, page)
    except Exception:
        logger.exception("Read page failed")
        return f"Failed to read page {page} from '{filename}'."


@mcp.tool()
def read_section(filename: str, section_title: str) -> str:
    """Read the full content of a named section from a document.

    Use this after get_sections to read a section you're interested in.
    Returns all text chunks for that section in document order, with page
    numbers.

    Get available section titles from get_sections first.

    Args:
        filename: The PDF filename (e.g., 'manual.pdf').
        section_title: Exact section title as returned by get_sections
            (e.g., 'Safety Guidelines').
    """
    try:
        settings = get_settings()
        chunks = get_section_chunks(filename, section_title, settings)
        return format_section_chunks(chunks, filename, section_title)
    except Exception:
        logger.exception("Read section failed")
        return f"Failed to read section '{section_title}' from '{filename}'."


@mcp.tool()
def list_docs() -> str:
    """List all ingested PDFs available for search.

    Start here to discover which documents are available. Returns filenames,
    chunk counts, and file hashes. Use the filenames with search_in_doc,
    get_sections, read_page, or read_section.
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
    """Get the table of contents (section headings) of a document.

    Use this to understand a document's structure before diving in.
    Returns numbered section titles in document order. Then use read_section
    to read any section, or search_in_doc to search within the document.

    Get valid filenames from list_docs first.

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


_SHUTDOWN_TIMEOUT = 5  # seconds to wait before forcing exit


def _cleanup() -> None:
    """Run server-side cleanup before exit."""
    logger.info("Running cleanup...")
    get_settings.cache_clear()
    logger.info("Cleanup complete")


async def _run_http(host: str, port: int) -> None:
    """Run the streamable-http transport with graceful shutdown."""
    import uvicorn

    app = mcp.streamable_http_app()
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=mcp.settings.log_level.lower(),
        timeout_graceful_shutdown=_SHUTDOWN_TIMEOUT,
    )
    server = uvicorn.Server(config)

    logger.info("Listening on http://%s:%d/mcp", host, port)
    await server.serve()
    _cleanup()
    logger.info("Server stopped")


async def _run_stdio() -> None:
    """Run the stdio transport with graceful shutdown."""
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Received signal — initiating graceful shutdown")
        shutdown_event.set()

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

    task = asyncio.create_task(mcp.run_stdio_async())

    done, pending = await asyncio.wait(
        [task, asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if task not in done:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=_SHUTDOWN_TIMEOUT)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            logger.info("Stdio task did not exit in time — forcing shutdown")

    for p in pending:
        p.cancel()
        try:
            await p
        except asyncio.CancelledError:
            pass

    _cleanup()
    logger.info("Server stopped")
    # stdio reads block the event loop and cannot be cancelled;
    # force-exit so the process doesn't hang.
    import os

    os._exit(0)


def run_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    name: str | None = None,
) -> None:
    """Start the MCP server with graceful shutdown support.

    Handles SIGINT and SIGTERM to drain connections (HTTP) or close
    streams (stdio) before exiting.

    Args:
        transport: Transport protocol ("stdio" or "streamable-http").
        host: Host to bind to (only used for HTTP transport).
        port: Port to bind to (only used for HTTP transport).
        name: Override the server name.
    """
    if name:
        mcp._mcp_server.name = name

    try:
        if transport == "stdio":
            asyncio.run(_run_stdio())
        else:
            mcp.settings.host = host
            mcp.settings.port = port
            asyncio.run(_run_http(host, port))
    except KeyboardInterrupt:
        pass
