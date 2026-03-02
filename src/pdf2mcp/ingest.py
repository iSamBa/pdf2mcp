"""Ingestion orchestrator: parsing → embedding → storage."""

from __future__ import annotations

import logging

from pdf2mcp.chunker import chunk_markdown
from pdf2mcp.config import Settings, get_settings
from pdf2mcp.embeddings import embed_texts
from pdf2mcp.parser import discover_pdfs, parse_pdf
from pdf2mcp.store import (
    clear_database,
    delete_by_source,
    get_db,
    get_ingested_files,
    record_ingestion,
    upsert_chunks,
)

__all__ = ["run_ingestion"]

logger = logging.getLogger(__name__)


def run_ingestion(
    settings: Settings | None = None,
    *,
    force: bool = False,
) -> None:
    """Run the full ingestion pipeline.

    Discovers PDFs, parses them, chunks the content, embeds chunks via
    OpenAI, and stores everything in LanceDB.

    Args:
        settings: Application settings. Uses ``get_settings()`` if None.
        force: If True, clear the database and re-ingest everything.
    """
    if settings is None:
        settings = get_settings()

    db = get_db(settings)

    if force:
        logger.info("Force mode: clearing existing database")
        clear_database(db)

    ingested = get_ingested_files(db)
    pdfs = discover_pdfs(settings.docs_dir)

    if not pdfs:
        logger.warning("No PDF files found in %s", settings.docs_dir)
        return

    ingested_count = 0
    skipped_count = 0

    for pdf_path in pdfs:
        filename = pdf_path.name

        # Parse
        try:
            parsed = parse_pdf(pdf_path)
        except Exception:
            logger.warning("Failed to parse %s, skipping", filename, exc_info=True)
            continue

        # Check if already ingested and unchanged
        if (
            not force
            and filename in ingested
            and ingested[filename] == parsed.file_hash
        ):
            logger.info("Skipping %s (unchanged)", filename)
            skipped_count += 1
            continue

        # If previously ingested but changed, delete old chunks
        if filename in ingested:
            logger.info("Re-ingesting %s (content changed)", filename)
            delete_by_source(db, filename)

        # Chunk
        chunks = chunk_markdown(
            parsed.markdown,
            filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            logger.warning("No chunks produced from %s", filename)
            continue

        # Embed
        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = embed_texts(texts, settings)
        except Exception:
            logger.warning(
                "Embedding failed for %s, skipping",
                filename,
                exc_info=True,
            )
            continue

        # Store
        try:
            upsert_chunks(db, chunks, embeddings, settings.embedding_dimensions)
            record_ingestion(db, filename, parsed.file_hash, len(chunks))
        except Exception:
            logger.warning(
                "Storing failed for %s, skipping",
                filename,
                exc_info=True,
            )
            continue

        ingested_count += 1
        logger.info("Ingested %s: %d chunks", filename, len(chunks))

    logger.info(
        "Ingestion complete: %d ingested, %d skipped",
        ingested_count,
        skipped_count,
    )
