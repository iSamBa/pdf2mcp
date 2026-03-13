"""LanceDB operations: create, upsert, delete, query."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import lancedb  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]

from pdf2mcp.config import ServerSettings
from pdf2mcp.models import DocumentChunk

__all__ = [
    "escape_filter_value",
    "table_exists",
    "get_db",
    "get_documents_table",
    "upsert_chunks",
    "delete_by_source",
    "get_ingested_files",
    "record_ingestion",
    "clear_database",
    "create_vector_index",
]

logger = logging.getLogger(__name__)

DOCUMENTS_TABLE = "documents"
METADATA_TABLE = "ingestion_metadata"

# Minimum rows needed for IVF_PQ index to be worthwhile
_MIN_ROWS_FOR_INDEX = 256


def escape_filter_value(value: str) -> str:
    """Escape a string value for use in a LanceDB filter predicate.

    LanceDB uses SQL-style escaping: single quotes are doubled (``''``).
    """
    return value.replace("'", "''")


def table_exists(db: lancedb.DBConnection, name: str) -> bool:
    """Check if a table exists in the database."""
    return name in db.list_tables().tables


@lru_cache(maxsize=8)
def _cached_connect(db_path_str: str) -> lancedb.DBConnection:
    """Return a cached DB connection for the given path."""
    return lancedb.connect(db_path_str)


def get_db(settings: ServerSettings) -> lancedb.DBConnection:
    """Open or create a LanceDB database in the configured data directory."""
    db_path = settings.data_dir / "lancedb"
    db_path.mkdir(parents=True, exist_ok=True)
    return _cached_connect(str(db_path))


# Cache for opened table handles: (db_path, table_name) -> Table
_table_cache: dict[tuple[str, str], lancedb.table.Table] = {}


def get_documents_table(settings: ServerSettings) -> lancedb.table.Table | None:
    """Return the opened documents table, or None if it doesn't exist or is empty.

    Caches the table handle so repeated calls skip list_tables/open_table.
    """
    db = get_db(settings)
    cache_key = (str(settings.data_dir / "lancedb"), DOCUMENTS_TABLE)

    if cache_key in _table_cache:
        return _table_cache[cache_key]

    if not table_exists(db, DOCUMENTS_TABLE):
        return None

    table = db.open_table(DOCUMENTS_TABLE)
    if table.count_rows() == 0:
        return None

    _table_cache[cache_key] = table
    return table


def invalidate_table_cache() -> None:
    """Clear the table cache (call after ingestion or clear_database)."""
    _table_cache.clear()


def _get_documents_schema(dimensions: int) -> pa.Schema:
    """Return the PyArrow schema for the documents table."""
    return pa.schema(
        [
            pa.field("text", pa.utf8()),
            pa.field(
                "vector",
                pa.list_(pa.float32(), list_size=dimensions),
            ),
            pa.field("source_file", pa.utf8()),
            pa.field("page_numbers", pa.list_(pa.int32())),
            pa.field("section_title", pa.utf8()),
            pa.field("chunk_index", pa.int32()),
        ]
    )


def _get_metadata_schema() -> pa.Schema:
    """Return the PyArrow schema for the ingestion metadata table."""
    return pa.schema(
        [
            pa.field("filename", pa.utf8()),
            pa.field("file_hash", pa.utf8()),
            pa.field("chunk_count", pa.int32()),
        ]
    )


def _ensure_documents_table(
    db: lancedb.DBConnection, dimensions: int
) -> lancedb.table.Table:
    """Create the documents table if it doesn't exist."""
    if table_exists(db, DOCUMENTS_TABLE):
        return db.open_table(DOCUMENTS_TABLE)
    schema = _get_documents_schema(dimensions)
    return db.create_table(DOCUMENTS_TABLE, schema=schema)


def _ensure_metadata_table(
    db: lancedb.DBConnection,
) -> lancedb.table.Table:
    """Create the metadata table if it doesn't exist."""
    if table_exists(db, METADATA_TABLE):
        return db.open_table(METADATA_TABLE)
    schema = _get_metadata_schema()
    return db.create_table(METADATA_TABLE, schema=schema)


def upsert_chunks(
    db: lancedb.DBConnection,
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    dimensions: int = 1536,
) -> None:
    """Insert chunks with their embeddings into the documents table."""
    if not chunks:
        return

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk/embedding count mismatch: "
            f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

    table = _ensure_documents_table(db, dimensions)
    records: list[dict[str, Any]] = []

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        records.append(
            {
                "text": chunk.text,
                "vector": embedding,
                "source_file": chunk.metadata.source_file,
                "page_numbers": chunk.metadata.page_numbers,
                "section_title": chunk.metadata.section_title,
                "chunk_index": chunk.metadata.chunk_index,
            }
        )

    table.add(records)
    invalidate_table_cache()
    logger.info("Stored %d chunks in documents table", len(records))


def delete_by_source(db: lancedb.DBConnection, source_file: str) -> None:
    """Delete all chunks from a specific source file."""
    if not table_exists(db, DOCUMENTS_TABLE):
        return
    table = db.open_table(DOCUMENTS_TABLE)
    escaped = escape_filter_value(source_file)
    table.delete(f"source_file = '{escaped}'")
    logger.info("Deleted chunks for source: %s", source_file)


def get_ingested_files(db: lancedb.DBConnection) -> dict[str, str]:
    """Return a mapping of filename to file_hash for all ingested files."""
    if not table_exists(db, METADATA_TABLE):
        return {}
    table = db.open_table(METADATA_TABLE)
    if table.count_rows() == 0:
        return {}
    arrow_table = table.to_arrow()
    filenames = arrow_table.column("filename").to_pylist()
    hashes = arrow_table.column("file_hash").to_pylist()
    return dict(zip(filenames, hashes, strict=True))


def record_ingestion(
    db: lancedb.DBConnection,
    filename: str,
    file_hash: str,
    chunk_count: int,
) -> None:
    """Record that a file has been ingested (insert or update)."""
    table = _ensure_metadata_table(db)

    # Delete existing record if present
    try:
        escaped = escape_filter_value(filename)
        table.delete(f"filename = '{escaped}'")
    except Exception:  # noqa: BLE001
        logger.debug("No existing record to delete for %s", filename, exc_info=True)

    table.add(
        [
            {
                "filename": filename,
                "file_hash": file_hash,
                "chunk_count": chunk_count,
            }
        ]
    )
    logger.info("Recorded ingestion: %s (%d chunks)", filename, chunk_count)


def clear_database(db: lancedb.DBConnection) -> None:
    """Drop all tables for a clean re-ingestion."""
    for name in [DOCUMENTS_TABLE, METADATA_TABLE]:
        if table_exists(db, name):
            db.drop_table(name)
    invalidate_table_cache()
    logger.info("Database cleared")


def create_vector_index(db: lancedb.DBConnection) -> None:
    """Create an IVF_PQ vector index on the documents table if it has enough rows.

    This speeds up vector search from brute-force kNN to approximate NN.
    Requires at least ``_MIN_ROWS_FOR_INDEX`` rows to be effective.
    """
    if not table_exists(db, DOCUMENTS_TABLE):
        return

    table = db.open_table(DOCUMENTS_TABLE)
    row_count = table.count_rows()

    if row_count < _MIN_ROWS_FOR_INDEX:
        logger.info(
            "Skipping vector index creation: %d rows < %d minimum",
            row_count,
            _MIN_ROWS_FOR_INDEX,
        )
        return

    try:
        table.create_index(metric="l2", index_type="IVF_PQ", replace=True)
        logger.info("Created vector index (IVF_PQ) on %d rows", row_count)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to create vector index", exc_info=True)
