"""Ingestion orchestrator: parsing → embedding → storage."""

from __future__ import annotations

import logging
from pathlib import Path

import lancedb

from pdf2mcp.chunker import chunk_markdown
from pdf2mcp.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSIONS,
    ServerSettings,
    get_settings,
)
from pdf2mcp.embeddings import compute_batch_count, embed_texts
from pdf2mcp.parser import discover_pdfs, parse_pdf
from pdf2mcp.progress import IngestionProgress
from pdf2mcp.store import (
    clear_database,
    create_fts_index,
    create_vector_index,
    delete_by_source,
    get_db,
    get_ingested_files,
    record_ingestion,
    upsert_chunks,
)

__all__ = ["run_ingestion"]

logger = logging.getLogger(__name__)


def run_ingestion(
    settings: ServerSettings | None = None,
    *,
    force: bool = False,
    show_progress: bool = False,
) -> None:
    """Run the full ingestion pipeline.

    Discovers PDFs, parses them, chunks the content, embeds chunks via
    OpenAI, and stores everything in LanceDB.

    Args:
        settings: Application settings. Uses ``get_settings()`` if None.
        force: If True, clear the database and re-ingest everything.
        show_progress: If True, display a Rich progress bar in the terminal.
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

    if show_progress:
        with IngestionProgress(total_docs=len(pdfs)) as progress:
            ingested_count, skipped_count = _ingest_pdfs(
                pdfs, db, settings, ingested, force, progress
            )
    else:
        ingested_count, skipped_count = _ingest_pdfs(
            pdfs, db, settings, ingested, force, None
        )

    # Create vector index for faster ANN search if enough rows exist
    if ingested_count > 0:
        create_vector_index(db)
        if settings.search_mode in ("hybrid", "keyword"):
            create_fts_index(db)

    logger.info(
        "Ingestion complete: %d ingested, %d skipped",
        ingested_count,
        skipped_count,
    )


def _ingest_pdfs(
    pdfs: list[Path],
    db: lancedb.DBConnection,
    settings: ServerSettings,
    ingested: dict[str, str],
    force: bool,
    progress: IngestionProgress | None,
) -> tuple[int, int]:
    """Process each PDF through the ingestion pipeline.

    Returns:
        Tuple of (ingested_count, skipped_count).
    """
    ingested_count = 0
    skipped_count = 0

    for pdf_path in pdfs:
        filename = pdf_path.name

        if progress is not None:
            progress.document_start(filename)

        # Parse (includes OCR if needed)
        if progress is not None:
            progress.stage_start("parsing")

        had_ocr = False

        def _on_ocr_start(total: int) -> None:
            nonlocal had_ocr
            had_ocr = True
            if progress is not None:
                progress.stage_complete()  # Complete parsing stage
                progress.set_ocr_pages(total)  # Start OCR stage

        def _on_ocr_page() -> None:
            if progress is not None:
                progress.advance_ocr()

        try:
            parsed = parse_pdf(
                pdf_path,
                ocr_enabled=settings.ocr_enabled,
                ocr_language=settings.ocr_language,
                ocr_dpi=settings.ocr_dpi,
                on_ocr_start=_on_ocr_start if progress is not None else None,
                on_ocr_page=_on_ocr_page if progress is not None else None,
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to parse %s, skipping", filename, exc_info=True)
            if progress is not None:
                progress.stage_complete()
                progress.document_complete()
            continue
        if progress is not None and not had_ocr:
            progress.stage_complete()

        # Check if already ingested and unchanged
        if (
            not force
            and filename in ingested
            and ingested[filename] == parsed.file_hash
        ):
            logger.info("Skipping %s (unchanged)", filename)
            skipped_count += 1
            if progress is not None:
                progress.document_skipped(filename)
            continue

        # If previously ingested but changed, delete old chunks
        if filename in ingested:
            logger.info("Re-ingesting %s (content changed)", filename)
            delete_by_source(db, filename)

        # Chunk
        if progress is not None:
            progress.stage_start("chunking")
        chunks = chunk_markdown(
            parsed.markdown,
            filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            logger.warning("No chunks produced from %s", filename)
            if progress is not None:
                progress.document_complete()
            continue
        if progress is not None:
            progress.stage_complete()

        # Embed — update progress total now that we know the batch count
        texts = [chunk.text for chunk in chunks]
        num_batches = compute_batch_count(len(texts), EMBEDDING_BATCH_SIZE)

        if progress is not None:
            progress.set_embedding_batches(num_batches)
            progress.stage_start("embedding")

        try:
            embeddings = embed_texts(
                texts,
                settings,
                on_batch_complete=progress.advance_embedding
                if progress is not None
                else None,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Embedding failed for %s, skipping",
                filename,
                exc_info=True,
            )
            if progress is not None:
                progress.document_complete()
            continue

        # Store
        if progress is not None:
            progress.stage_start("storing")
        try:
            upsert_chunks(db, chunks, embeddings, EMBEDDING_DIMENSIONS)
            record_ingestion(db, filename, parsed.file_hash, len(chunks))
        except Exception:  # noqa: BLE001
            logger.warning(
                "Storing failed for %s, skipping",
                filename,
                exc_info=True,
            )
            if progress is not None:
                progress.document_complete()
            continue
        if progress is not None:
            progress.stage_complete()

        if progress is not None:
            progress.document_complete()

        ingested_count += 1
        logger.info("Ingested %s: %d chunks", filename, len(chunks))

    return ingested_count, skipped_count
