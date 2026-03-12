"""Tests for pdf2mcp.search module."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.models import ChunkMetadata, DocumentChunk
from pdf2mcp.search import (
    SearchResult,
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
from pdf2mcp.store import get_db, record_ingestion, upsert_chunks

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def settings(tmp_path: Path) -> MagicMock:
    """Create mock settings with a temporary data directory."""
    s = MagicMock()
    s.data_dir = tmp_path / "data"
    s.data_dir.mkdir()
    s.openai_api_key.get_secret_value.return_value = "sk-test"
    s.embedding_model = "text-embedding-3-small"
    s.embedding_dimensions = 8
    s.default_num_results = 5
    return s


@pytest.fixture()
def db(settings: MagicMock):  # type: ignore[no-untyped-def]
    """Create a fresh LanceDB connection."""
    connection = get_db(settings)
    yield connection
    db_path = settings.data_dir / "lancedb"
    if db_path.exists():
        shutil.rmtree(db_path)


def _make_chunks(
    source_file: str = "test.pdf",
    count: int = 3,
    section_prefix: str = "Section",
) -> list[DocumentChunk]:
    """Create a list of test DocumentChunks."""
    return [
        DocumentChunk(
            text=f"Chunk {i} content from {source_file}",
            metadata=ChunkMetadata(
                source_file=source_file,
                page_numbers=[i + 1],
                section_title=f"{section_prefix} {i}",
                chunk_index=i,
            ),
        )
        for i in range(count)
    ]


def _make_embeddings(count: int = 3, dim: int = 8) -> list[list[float]]:
    """Create fake embeddings."""
    return [[float(i) / 10.0] * dim for i in range(count)]


def _populate_db(
    db: MagicMock,
    source_file: str = "test.pdf",
    count: int = 3,
    dim: int = 8,
) -> None:
    """Insert test chunks and metadata into the database."""
    chunks = _make_chunks(source_file=source_file, count=count)
    embeddings = _make_embeddings(count=count, dim=dim)
    upsert_chunks(db, chunks, embeddings, dimensions=dim)
    record_ingestion(db, source_file, "hash123", count)


# ── SearchResult ────────────────────────────────────────────────────


class TestSearchResult:
    """Test the SearchResult model."""

    def test_creates_valid_result(self) -> None:
        result = SearchResult(
            text="test text",
            score=0.5,
            source_file="test.pdf",
            page_numbers=[1, 2],
            section_title="Section A",
        )
        assert result.text == "test text"
        assert result.score == 0.5
        assert result.page_numbers == [1, 2]


# ── search_documents ───────────────────────────────────────────────


class TestSearchDocuments:
    """Test the search_documents function."""

    def test_empty_query_returns_empty(self, settings: MagicMock) -> None:
        assert search_documents("", settings) == []

    def test_whitespace_query_returns_empty(self, settings: MagicMock) -> None:
        assert search_documents("   ", settings) == []

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_empty_database_returns_empty(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        results = search_documents("test query", settings)
        assert results == []

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_returns_results_with_correct_structure(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)

        results = search_documents("test query", settings, num_results=2)

        assert len(results) == 2
        for result in results:
            assert isinstance(result, SearchResult)
            assert result.source_file == "manual.pdf"
            assert isinstance(result.text, str)
            assert isinstance(result.score, float)
            assert isinstance(result.page_numbers, list)
            assert isinstance(result.section_title, str)

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_respects_num_results(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, source_file="test.pdf", count=5)

        results = search_documents("query", settings, num_results=2)
        assert len(results) == 2

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_uses_default_num_results(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, source_file="test.pdf", count=10)
        settings.default_num_results = 3

        results = search_documents("query", settings)
        assert len(results) == 3

    @patch("pdf2mcp.search.embed_texts")
    def test_strips_query_whitespace(
        self, mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        mock_embed.return_value = [[0.1] * 8]
        _populate_db(db, count=1)

        search_documents("  hello world  ", settings)
        mock_embed.assert_called_once_with(["hello world"], settings)

    @patch("pdf2mcp.search.embed_texts")
    def test_returns_empty_on_embedding_failure(
        self, mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, count=1)
        mock_embed.side_effect = RuntimeError("API error")

        results = search_documents("query", settings)
        assert results == []


# ── format_results ─────────────────────────────────────────────────


class TestFormatResults:
    """Test result formatting."""

    def test_empty_results(self) -> None:
        output = format_results([])
        assert "No results found" in output
        assert "ingested" in output

    def test_formats_single_result(self) -> None:
        results = [
            SearchResult(
                text="PDF document specs",
                score=0.1234,
                source_file="manual.pdf",
                page_numbers=[5, 6],
                section_title="Specifications",
            )
        ]
        output = format_results(results)
        assert "Result 1" in output
        assert "0.1234" in output
        assert "manual.pdf" in output
        assert "Specifications" in output
        assert "pages 5, 6" in output
        assert "PDF document specs" in output

    def test_formats_multiple_results(self) -> None:
        results = [
            SearchResult(
                text="First result",
                score=0.1,
                source_file="a.pdf",
                page_numbers=[1],
                section_title="Intro",
            ),
            SearchResult(
                text="Second result",
                score=0.2,
                source_file="b.pdf",
                page_numbers=[2],
                section_title="Details",
            ),
        ]
        output = format_results(results)
        assert "Result 1" in output
        assert "Result 2" in output
        assert "First result" in output
        assert "Second result" in output

    def test_handles_missing_section_title(self) -> None:
        results = [
            SearchResult(
                text="content",
                score=0.5,
                source_file="test.pdf",
                page_numbers=[1],
                section_title="",
            )
        ]
        output = format_results(results)
        assert "test.pdf" in output
        # No ">" separator when section title is empty
        assert " > " not in output

    def test_handles_empty_page_numbers(self) -> None:
        results = [
            SearchResult(
                text="content",
                score=0.5,
                source_file="test.pdf",
                page_numbers=[],
                section_title="Section",
            )
        ]
        output = format_results(results)
        assert "pages" not in output


# ── list_ingested_documents ─────────────────────────────────────────


class TestListIngestedDocuments:
    """Test document listing."""

    def test_empty_database(self, settings: MagicMock, db: MagicMock) -> None:
        result = list_ingested_documents(settings)
        assert result == []

    def test_returns_metadata(self, settings: MagicMock, db: MagicMock) -> None:
        record_ingestion(db, "manual.pdf", "hash_a", 10)
        record_ingestion(db, "guide.pdf", "hash_b", 5)

        result = list_ingested_documents(settings)
        assert len(result) == 2

        filenames = {doc["filename"] for doc in result}
        assert filenames == {"manual.pdf", "guide.pdf"}

        for doc in result:
            assert "filename" in doc
            assert "file_hash" in doc
            assert "chunk_count" in doc

    def test_returns_correct_values(self, settings: MagicMock, db: MagicMock) -> None:
        record_ingestion(db, "test.pdf", "abc123", 42)

        result = list_ingested_documents(settings)
        assert len(result) == 1
        assert result[0]["filename"] == "test.pdf"
        assert result[0]["file_hash"] == "abc123"
        assert result[0]["chunk_count"] == 42


# ── get_document_sections ──────────────────────────────────────────


class TestGetDocumentSections:
    """Test section retrieval."""

    def test_empty_database(self, settings: MagicMock, db: MagicMock) -> None:
        result = get_document_sections("test.pdf", settings)
        assert result == []

    def test_unknown_filename(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="known.pdf", count=2)
        result = get_document_sections("unknown.pdf", settings)
        assert result == []

    def test_returns_unique_sections(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)

        sections = get_document_sections("manual.pdf", settings)
        assert len(sections) == 3
        assert sections == ["Section 0", "Section 1", "Section 2"]

    def test_preserves_document_order(self, settings: MagicMock, db: MagicMock) -> None:
        # Create chunks with non-sequential section titles
        chunks = [
            DocumentChunk(
                text=f"Chunk {i}",
                metadata=ChunkMetadata(
                    source_file="manual.pdf",
                    page_numbers=[i],
                    section_title=title,
                    chunk_index=i,
                ),
            )
            for i, title in enumerate(["Intro", "Specs", "Intro", "Safety"])
        ]
        embeddings = _make_embeddings(count=4)
        upsert_chunks(db, chunks, embeddings, dimensions=8)

        sections = get_document_sections("manual.pdf", settings)
        # "Intro" appears at chunk_index 0 and 2, should only appear once (first)
        assert sections == ["Intro", "Specs", "Safety"]

    def test_handles_single_quote_in_filename(
        self, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, source_file="o'reilly.pdf", count=2)
        sections = get_document_sections("o'reilly.pdf", settings)
        assert len(sections) == 2

    def test_skips_empty_section_titles(
        self, settings: MagicMock, db: MagicMock
    ) -> None:
        chunks = [
            DocumentChunk(
                text="Chunk 0",
                metadata=ChunkMetadata(
                    source_file="test.pdf",
                    page_numbers=[1],
                    section_title="",
                    chunk_index=0,
                ),
            ),
            DocumentChunk(
                text="Chunk 1",
                metadata=ChunkMetadata(
                    source_file="test.pdf",
                    page_numbers=[2],
                    section_title="Real Section",
                    chunk_index=1,
                ),
            ),
        ]
        embeddings = _make_embeddings(count=2)
        upsert_chunks(db, chunks, embeddings, dimensions=8)

        sections = get_document_sections("test.pdf", settings)
        assert sections == ["Real Section"]


# ── search_in_document ────────────────────────────────────────────


class TestSearchInDocument:
    """Test the search_in_document function."""

    def test_empty_query_returns_empty(self, settings: MagicMock) -> None:
        assert search_in_document("", "test.pdf", settings) == []

    def test_whitespace_query_returns_empty(self, settings: MagicMock) -> None:
        assert search_in_document("   ", "test.pdf", settings) == []

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_returns_only_matching_document(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)
        _populate_db(db, source_file="guide.pdf", count=3)

        results = search_in_document("test query", "manual.pdf", settings, num_results=5)

        assert len(results) > 0
        for result in results:
            assert result.source_file == "manual.pdf"

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_unknown_file_returns_empty(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)

        results = search_in_document("test query", "unknown.pdf", settings)
        assert results == []

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_empty_database_returns_empty(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        results = search_in_document("test query", "test.pdf", settings)
        assert results == []

    @patch("pdf2mcp.search.embed_texts")
    def test_returns_empty_on_embedding_failure(
        self, mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, count=1)
        mock_embed.side_effect = RuntimeError("API error")

        results = search_in_document("query", "test.pdf", settings)
        assert results == []

    @patch("pdf2mcp.search.embed_texts", return_value=[[0.1] * 8])
    def test_respects_num_results(
        self, _mock_embed: MagicMock, settings: MagicMock, db: MagicMock
    ) -> None:
        _populate_db(db, source_file="test.pdf", count=5)

        results = search_in_document("query", "test.pdf", settings, num_results=2)
        assert len(results) == 2


# ── get_page_chunks ───────────────────────────────────────────────


class TestGetPageChunks:
    """Test the get_page_chunks function."""

    def test_empty_database(self, settings: MagicMock, db: MagicMock) -> None:
        result = get_page_chunks("test.pdf", 1, settings)
        assert result == []

    def test_unknown_filename(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="known.pdf", count=3)
        result = get_page_chunks("unknown.pdf", 1, settings)
        assert result == []

    def test_returns_matching_page(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)

        # _make_chunks assigns page_numbers=[i+1], so page 2 -> chunk_index 1
        result = get_page_chunks("manual.pdf", 2, settings)
        assert len(result) == 1
        assert result[0]["chunk_index"] == 1
        assert 2 in result[0]["page_numbers"]

    def test_returns_multiple_chunks_for_same_page(
        self, settings: MagicMock, db: MagicMock
    ) -> None:
        chunks = [
            DocumentChunk(
                text=f"Chunk {i}",
                metadata=ChunkMetadata(
                    source_file="manual.pdf",
                    page_numbers=[5],
                    section_title="Section A",
                    chunk_index=i,
                ),
            )
            for i in range(3)
        ]
        embeddings = _make_embeddings(count=3)
        upsert_chunks(db, chunks, embeddings, dimensions=8)

        result = get_page_chunks("manual.pdf", 5, settings)
        assert len(result) == 3
        # Verify ordered by chunk_index
        indices = [r["chunk_index"] for r in result]
        assert indices == sorted(indices)

    def test_page_not_found(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)
        result = get_page_chunks("manual.pdf", 999, settings)
        assert result == []


# ── get_section_chunks ────────────────────────────────────────────


class TestGetSectionChunks:
    """Test the get_section_chunks function."""

    def test_empty_database(self, settings: MagicMock, db: MagicMock) -> None:
        result = get_section_chunks("test.pdf", "Intro", settings)
        assert result == []

    def test_unknown_filename(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="known.pdf", count=3)
        result = get_section_chunks("unknown.pdf", "Section 0", settings)
        assert result == []

    def test_returns_matching_section(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)

        result = get_section_chunks("manual.pdf", "Section 1", settings)
        assert len(result) == 1
        assert result[0]["section_title"] == "Section 1"

    def test_returns_multiple_chunks_for_same_section(
        self, settings: MagicMock, db: MagicMock
    ) -> None:
        chunks = [
            DocumentChunk(
                text=f"Chunk {i}",
                metadata=ChunkMetadata(
                    source_file="manual.pdf",
                    page_numbers=[i + 1],
                    section_title="Safety",
                    chunk_index=i,
                ),
            )
            for i in range(3)
        ]
        embeddings = _make_embeddings(count=3)
        upsert_chunks(db, chunks, embeddings, dimensions=8)

        result = get_section_chunks("manual.pdf", "Safety", settings)
        assert len(result) == 3
        indices = [r["chunk_index"] for r in result]
        assert indices == sorted(indices)

    def test_section_not_found(self, settings: MagicMock, db: MagicMock) -> None:
        _populate_db(db, source_file="manual.pdf", count=3)
        result = get_section_chunks("manual.pdf", "Nonexistent", settings)
        assert result == []

    def test_handles_single_quote_in_section_title(
        self, settings: MagicMock, db: MagicMock
    ) -> None:
        chunks = [
            DocumentChunk(
                text="Content",
                metadata=ChunkMetadata(
                    source_file="manual.pdf",
                    page_numbers=[1],
                    section_title="What's New",
                    chunk_index=0,
                ),
            )
        ]
        embeddings = _make_embeddings(count=1)
        upsert_chunks(db, chunks, embeddings, dimensions=8)

        result = get_section_chunks("manual.pdf", "What's New", settings)
        assert len(result) == 1


# ── format_page_chunks ────────────────────────────────────────────


class TestFormatPageChunks:
    """Test page chunk formatting."""

    def test_empty_chunks(self) -> None:
        output = format_page_chunks([], "test.pdf", 5)
        assert "No content found" in output
        assert "page 5" in output
        assert "test.pdf" in output

    def test_formats_with_header(self) -> None:
        chunks = [
            {"text": "Hello world", "section_title": "Intro", "chunk_index": 0,
             "source_file": "test.pdf", "page_numbers": [1]},
        ]
        output = format_page_chunks(chunks, "test.pdf", 1)
        assert "Page 1 of test.pdf" in output
        assert "Hello world" in output
        assert "## Intro" in output


# ── format_section_chunks ─────────────────────────────────────────


class TestFormatSectionChunks:
    """Test section chunk formatting."""

    def test_empty_chunks(self) -> None:
        output = format_section_chunks([], "test.pdf", "Intro")
        assert "No content found" in output
        assert "Intro" in output
        assert "test.pdf" in output

    def test_formats_with_header_and_pages(self) -> None:
        chunks = [
            {"text": "First chunk", "section_title": "Safety", "chunk_index": 0,
             "source_file": "manual.pdf", "page_numbers": [3, 4]},
            {"text": "Second chunk", "section_title": "Safety", "chunk_index": 1,
             "source_file": "manual.pdf", "page_numbers": [4, 5]},
        ]
        output = format_section_chunks(chunks, "manual.pdf", "Safety")
        assert "Section 'Safety' from manual.pdf" in output
        assert "pages 3, 4, 5" in output
        assert "First chunk" in output
        assert "Second chunk" in output
