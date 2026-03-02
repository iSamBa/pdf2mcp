"""Search query logic and result formatting."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from pdf2mcp.config import Settings
from pdf2mcp.embeddings import embed_texts
from pdf2mcp.store import DOCUMENTS_TABLE, METADATA_TABLE, _escape_filter_value, get_db

__all__ = [
    "SearchResult",
    "format_results",
    "get_document_sections",
    "list_ingested_documents",
    "search_documents",
]

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """A single search result."""

    text: str
    score: float
    source_file: str
    page_numbers: list[int]
    section_title: str


def search_documents(
    query: str,
    settings: Settings,
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

    # Embed the query (reuses retry logic from embeddings module)
    try:
        vectors = embed_texts([query.strip()], settings)
    except Exception:
        logger.warning("Failed to embed search query", exc_info=True)
        return []
    query_vector = vectors[0]

    # Search LanceDB
    db = get_db(settings)
    if DOCUMENTS_TABLE not in db.list_tables().tables:
        logger.warning("No documents table found. Run ingestion first.")
        return []

    table = db.open_table(DOCUMENTS_TABLE)
    if table.count_rows() == 0:
        logger.warning("Documents table is empty. Run ingestion first.")
        return []

    rows: list[dict[str, Any]] = table.search(query_vector).limit(num_results).to_list()

    return [
        SearchResult(
            text=row["text"],
            score=row.get("_distance", 0.0),
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


def list_ingested_documents(settings: Settings) -> list[dict[str, Any]]:
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


def get_document_sections(filename: str, settings: Settings) -> list[str]:
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
