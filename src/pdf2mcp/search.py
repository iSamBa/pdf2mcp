"""Search query logic and result formatting."""

from __future__ import annotations

import logging
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from pydantic import BaseModel

from pdf2mcp.config import ServerSettings
from pdf2mcp.embeddings import embed_query
from pdf2mcp.store import (
    DOCUMENTS_TABLE,
    METADATA_TABLE,
    _escape_filter_value,
    get_db,
    get_documents_table,
)

__all__ = [
    "SearchResult",
    "format_page_chunks",
    "format_results",
    "format_section_chunks",
    "get_document_sections",
    "get_page_chunks",
    "get_section_chunks",
    "list_ingested_documents",
    "search_documents",
    "search_in_document",
]

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """A single search result."""

    text: str
    score: float
    source_file: str
    page_numbers: list[int]
    section_title: str


def _distance_to_similarity(distance: float) -> float:
    """Convert L2 distance to a 0-1 similarity score (higher = more relevant)."""
    return 1.0 / (1.0 + distance)


def search_documents(
    query: str,
    settings: ServerSettings,
    *,
    num_results: int | None = None,
) -> list[SearchResult]:
    """Search ingested documents for relevant passages.

    Embeds the query using the configured OpenAI model and performs
    a vector similarity search in LanceDB.

    Args:
        query: Natural language search query.
        settings: Application settings.
        num_results: Number of results to return. Falls back to
            ``settings.default_num_results`` when *None*.

    Returns:
        Ordered list of search results (most relevant first).
    """
    if not query or not query.strip():
        return []

    if num_results is None:
        num_results = settings.default_num_results

    # Embed the query with caching
    query_vector = embed_query(query.strip(), settings)
    if query_vector is None:
        return []

    # Search LanceDB using cached table handle
    table = get_documents_table(settings)
    if table is None:
        logger.warning(
            "No documents table found or table is empty."
            " Run ingestion first."
        )
        return []

    rows: list[dict[str, Any]] = table.search(query_vector).limit(num_results).to_list()

    return [
        SearchResult(
            text=row["text"],
            score=_distance_to_similarity(row.get("_distance", 0.0)),
            source_file=row["source_file"],
            page_numbers=row.get("page_numbers", []),
            section_title=row.get("section_title", ""),
        )
        for row in rows
    ]


def format_results(results: list[SearchResult]) -> str:
    """Format search results for LLM consumption.

    Produces clean, structured text with source provenance that an LLM
    can easily parse and cite.
    """
    if not results:
        return "No results found. Make sure documents have been ingested."

    parts: list[str] = []
    for i, result in enumerate(results, 1):
        source_info = f"Source: {result.source_file}"
        if result.section_title:
            source_info += f" > {result.section_title}"
        if result.page_numbers:
            pages = ", ".join(str(p) for p in result.page_numbers)
            source_info += f" (pages {pages})"

        parts.append(
            f"--- Result {i} (score: {result.score:.4f}) ---\n"
            f"{source_info}\n\n"
            f"{result.text}"
        )

    return "\n\n".join(parts)


def list_ingested_documents(settings: ServerSettings) -> list[dict[str, Any]]:
    """List all ingested documents with metadata.

    Returns:
        List of dicts with keys: filename, file_hash, chunk_count.
    """
    db = get_db(settings)
    if METADATA_TABLE not in db.list_tables().tables:
        return []

    table = db.open_table(METADATA_TABLE)
    if table.count_rows() == 0:
        return []

    arrow_table = table.to_arrow()
    filenames = arrow_table.column("filename").to_pylist()
    hashes = arrow_table.column("file_hash").to_pylist()
    counts = arrow_table.column("chunk_count").to_pylist()

    return [
        {"filename": fn, "file_hash": fh, "chunk_count": cc}
        for fn, fh, cc in zip(filenames, hashes, counts)
    ]


def get_document_sections(filename: str, settings: ServerSettings) -> list[str]:
    """Get unique section titles for a specific document.

    Returns:
        Ordered list of unique section titles (preserving first-occurrence order).
    """
    db = get_db(settings)
    if DOCUMENTS_TABLE not in db.list_tables().tables:
        return []

    table = db.open_table(DOCUMENTS_TABLE)
    escaped = _escape_filter_value(filename)
    arrow_table = (
        table.search()
        .where(f"source_file = '{escaped}'")
        .select(["section_title", "chunk_index"])
        .to_arrow()
    )

    if arrow_table.num_rows == 0:
        return []

    titles = arrow_table.column("section_title").to_pylist()
    indices = arrow_table.column("chunk_index").to_pylist()

    # Sort by chunk_index to preserve document order, then deduplicate
    paired = sorted(zip(indices, titles))
    seen: set[str] = set()
    sections: list[str] = []
    for _, title in paired:
        if title and title not in seen:
            seen.add(title)
            sections.append(title)

    return sections


def search_in_document(
    query: str,
    filename: str,
    settings: ServerSettings,
    *,
    num_results: int | None = None,
) -> list[SearchResult]:
    """Search within a single document by filtering on source_file.

    Args:
        query: Natural language search query.
        filename: The PDF filename to scope the search to.
        settings: Application settings.
        num_results: Number of results to return.

    Returns:
        Ordered list of search results from the specified document.
    """
    if not query or not query.strip():
        return []

    if num_results is None:
        num_results = settings.default_num_results

    # Embed the query with caching
    query_vector = embed_query(query.strip(), settings)
    if query_vector is None:
        return []

    # Use cached table handle
    table = get_documents_table(settings)
    if table is None:
        logger.warning(
            "No documents table found or table is empty."
            " Run ingestion first."
        )
        return []

    escaped = _escape_filter_value(filename)
    rows: list[dict[str, Any]] = (
        table.search(query_vector)
        .where(f"source_file = '{escaped}'")
        .limit(num_results)
        .to_list()
    )

    return [
        SearchResult(
            text=row["text"],
            score=_distance_to_similarity(row.get("_distance", 0.0)),
            source_file=row["source_file"],
            page_numbers=row.get("page_numbers", []),
            section_title=row.get("section_title", ""),
        )
        for row in rows
    ]


def get_page_chunks(
    filename: str,
    page: int,
    settings: ServerSettings,
) -> list[dict[str, Any]]:
    """Return all chunks whose page_numbers contain the given page.

    Results are ordered by chunk_index. This is a pure metadata query
    (no embeddings needed). Uses Arrow compute to filter server-side
    instead of converting the full table to Python.

    Args:
        filename: The PDF filename.
        page: The page number to retrieve.
        settings: Application settings.

    Returns:
        List of dicts with keys: text, source_file, page_numbers,
        section_title, chunk_index.
    """
    db = get_db(settings)
    if DOCUMENTS_TABLE not in db.list_tables().tables:
        return []

    table = db.open_table(DOCUMENTS_TABLE)
    escaped = _escape_filter_value(filename)
    arrow_table = (
        table.search()
        .where(f"source_file = '{escaped}'")
        .select(["text", "source_file", "page_numbers", "section_title", "chunk_index"])
        .to_arrow()
    )

    if arrow_table.num_rows == 0:
        return []

    # Filter using Arrow: build a boolean mask without converting entire table to Python
    page_col = arrow_table.column("page_numbers")
    row_mask = pa.array(
        [page in page_col[i].as_py() for i in range(arrow_table.num_rows)],
        type=pa.bool_(),
    )
    filtered = arrow_table.filter(row_mask)

    if filtered.num_rows == 0:
        return []

    # Sort by chunk_index using Arrow compute
    sort_indices = pc.sort_indices(filtered, sort_keys=[("chunk_index", "ascending")])
    filtered = filtered.take(sort_indices)

    # Convert columnar dict to list of row dicts
    col_dict = filtered.to_pydict()
    return [
        {col: col_dict[col][i] for col in col_dict}
        for i in range(filtered.num_rows)
    ]


def get_section_chunks(
    filename: str,
    section_title: str,
    settings: ServerSettings,
) -> list[dict[str, Any]]:
    """Return all chunks matching a source_file and section_title.

    Results are ordered by chunk_index. This is a pure metadata query
    (no embeddings needed).

    Args:
        filename: The PDF filename.
        section_title: The section title to retrieve.
        settings: Application settings.

    Returns:
        List of dicts with keys: text, source_file, page_numbers,
        section_title, chunk_index.
    """
    db = get_db(settings)
    if DOCUMENTS_TABLE not in db.list_tables().tables:
        return []

    table = db.open_table(DOCUMENTS_TABLE)
    escaped_file = _escape_filter_value(filename)
    escaped_section = _escape_filter_value(section_title)
    arrow_table = (
        table.search()
        .where(
            f"source_file = '{escaped_file}' AND section_title = '{escaped_section}'"
        )
        .select(["text", "source_file", "page_numbers", "section_title", "chunk_index"])
        .to_arrow()
    )

    if arrow_table.num_rows == 0:
        return []

    rows = arrow_table.to_pydict()
    results = []
    for i in range(arrow_table.num_rows):
        results.append(
            {
                "text": rows["text"][i],
                "source_file": rows["source_file"][i],
                "page_numbers": rows["page_numbers"][i],
                "section_title": rows["section_title"][i],
                "chunk_index": rows["chunk_index"][i],
            }
        )

    results.sort(key=lambda r: r["chunk_index"])
    return results


def format_page_chunks(chunks: list[dict[str, Any]], filename: str, page: int) -> str:
    """Format page chunks for LLM consumption.

    Joins chunk texts in document order with a metadata header.
    """
    if not chunks:
        return (
            f"No content found for page {page} in '{filename}'. "
            "Check the filename with list_docs."
        )

    sections = []
    current_section = None
    for chunk in chunks:
        title = chunk.get("section_title", "")
        if title != current_section:
            current_section = title
            if title:
                sections.append(f"\n## {title}\n")

        sections.append(chunk["text"])

    header = f"Page {page} of {filename}:\n"
    return header + "\n".join(sections)


def format_section_chunks(
    chunks: list[dict[str, Any]], filename: str, section_title: str
) -> str:
    """Format section chunks for LLM consumption.

    Joins chunk texts in document order with a metadata header.
    """
    if not chunks:
        return (
            f"No content found for section '{section_title}' in '{filename}'. "
            "Check available sections with get_sections."
        )

    pages: set[int] = set()
    for chunk in chunks:
        pages.update(chunk.get("page_numbers", []))

    sorted_pages = sorted(pages)
    page_info = ", ".join(str(p) for p in sorted_pages) if sorted_pages else "unknown"

    header = f"Section '{section_title}' from {filename} (pages {page_info}):\n\n"
    body = "\n\n".join(chunk["text"] for chunk in chunks)
    return header + body
