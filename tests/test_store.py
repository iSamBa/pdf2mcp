"""Tests for pdf2mcp.store module."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pdf2mcp.models import ChunkMetadata, DocumentChunk
from pdf2mcp.store import (
    clear_database,
    delete_by_source,
    get_db,
    get_ingested_files,
    record_ingestion,
    upsert_chunks,
)


@pytest.fixture()
def settings(tmp_path: Path) -> MagicMock:
    """Create mock settings with a temporary data directory."""
    s = MagicMock()
    s.data_dir = tmp_path / "data"
    s.data_dir.mkdir()
    return s


@pytest.fixture()
def db(settings: MagicMock):  # type: ignore[no-untyped-def]
    """Create a fresh LanceDB connection."""
    connection = get_db(settings)
    yield connection
    # Cleanup
    db_path = settings.data_dir / "lancedb"
    if db_path.exists():
        shutil.rmtree(db_path)


def _make_chunks(
    source_file: str = "test.pdf",
    count: int = 3,
) -> list[DocumentChunk]:
    """Create a list of test DocumentChunks."""
    return [
        DocumentChunk(
            text=f"Chunk {i} content from {source_file}",
            metadata=ChunkMetadata(
                source_file=source_file,
                page_numbers=[i + 1],
                section_title=f"Section {i}",
                chunk_index=i,
            ),
        )
        for i in range(count)
    ]


def _make_embeddings(count: int = 3, dim: int = 8) -> list[list[float]]:
    """Create fake embeddings."""
    return [[float(i) / 10.0] * dim for i in range(count)]


class TestGetDb:
    """Test database connection."""

    def test_creates_db_directory(self, settings: MagicMock) -> None:
        get_db(settings)
        assert (settings.data_dir / "lancedb").exists()

    def test_returns_connection(self, settings: MagicMock) -> None:
        db = get_db(settings)
        assert db is not None


class TestUpsertChunks:
    """Test chunk storage."""

    def test_stores_chunks(self, db):  # type: ignore[no-untyped-def]
        chunks = _make_chunks(count=3)
        embeddings = _make_embeddings(count=3)

        upsert_chunks(db, chunks, embeddings, dimensions=8)

        table = db.open_table("documents")
        assert table.count_rows() == 3

    def test_stores_metadata_correctly(self, db):  # type: ignore[no-untyped-def]
        chunks = _make_chunks(source_file="manual.pdf", count=1)
        embeddings = _make_embeddings(count=1)

        upsert_chunks(db, chunks, embeddings, dimensions=8)

        table = db.open_table("documents")
        arrow = table.to_arrow()
        assert arrow.column("source_file")[0].as_py() == "manual.pdf"
        assert arrow.column("section_title")[0].as_py() == "Section 0"
        assert arrow.column("chunk_index")[0].as_py() == 0

    def test_empty_chunks_noop(self, db):  # type: ignore[no-untyped-def]
        upsert_chunks(db, [], [], dimensions=8)
        assert "documents" not in db.list_tables().tables

    def test_adds_to_existing_table(self, db):  # type: ignore[no-untyped-def]
        chunks1 = _make_chunks(source_file="a.pdf", count=2)
        chunks2 = _make_chunks(source_file="b.pdf", count=3)
        embeddings1 = _make_embeddings(count=2)
        embeddings2 = _make_embeddings(count=3)

        upsert_chunks(db, chunks1, embeddings1, dimensions=8)
        upsert_chunks(db, chunks2, embeddings2, dimensions=8)

        table = db.open_table("documents")
        assert table.count_rows() == 5


class TestDeleteBySource:
    """Test deletion by source file."""

    def test_deletes_matching_chunks(self, db):  # type: ignore[no-untyped-def]
        chunks_a = _make_chunks(source_file="a.pdf", count=2)
        chunks_b = _make_chunks(source_file="b.pdf", count=3)
        emb_a = _make_embeddings(count=2)
        emb_b = _make_embeddings(count=3)

        upsert_chunks(db, chunks_a, emb_a, dimensions=8)
        upsert_chunks(db, chunks_b, emb_b, dimensions=8)

        delete_by_source(db, "a.pdf")

        table = db.open_table("documents")
        assert table.count_rows() == 3
        sources = table.to_arrow().column("source_file").to_pylist()
        assert all(s == "b.pdf" for s in sources)

    def test_noop_on_missing_table(self, db):  # type: ignore[no-untyped-def]
        # Should not raise
        delete_by_source(db, "nonexistent.pdf")


class TestIngestionMetadata:
    """Test file ingestion tracking."""

    def test_record_and_retrieve(self, db):  # type: ignore[no-untyped-def]
        record_ingestion(db, "test.pdf", "abc123", 10)

        ingested = get_ingested_files(db)
        assert ingested == {"test.pdf": "abc123"}

    def test_multiple_records(self, db):  # type: ignore[no-untyped-def]
        record_ingestion(db, "a.pdf", "hash_a", 5)
        record_ingestion(db, "b.pdf", "hash_b", 10)

        ingested = get_ingested_files(db)
        assert ingested == {"a.pdf": "hash_a", "b.pdf": "hash_b"}

    def test_update_existing_record(self, db):  # type: ignore[no-untyped-def]
        record_ingestion(db, "test.pdf", "old_hash", 5)
        record_ingestion(db, "test.pdf", "new_hash", 8)

        ingested = get_ingested_files(db)
        assert ingested["test.pdf"] == "new_hash"

    def test_empty_database(self, db):  # type: ignore[no-untyped-def]
        ingested = get_ingested_files(db)
        assert ingested == {}


class TestClearDatabase:
    """Test database clearing."""

    def test_clears_all_tables(self, db):  # type: ignore[no-untyped-def]
        chunks = _make_chunks(count=2)
        embeddings = _make_embeddings(count=2)
        upsert_chunks(db, chunks, embeddings, dimensions=8)
        record_ingestion(db, "test.pdf", "hash", 2)

        clear_database(db)

        assert "documents" not in db.list_tables().tables
        assert "ingestion_metadata" not in db.list_tables().tables

    def test_clears_empty_database(self, db):  # type: ignore[no-untyped-def]
        # Should not raise
        clear_database(db)
