"""Tests for pdf2mcp.models module."""

import pytest
from pydantic import ValidationError

from pdf2mcp.models import ChunkMetadata, DocumentChunk, ParsedDocument


class TestChunkMetadata:
    """Test ChunkMetadata model validation."""

    def test_valid_metadata(self) -> None:
        meta = ChunkMetadata(
            source_file="manual.pdf",
            page_numbers=[1, 2],
            section_title="Introduction",
            chunk_index=0,
        )
        assert meta.source_file == "manual.pdf"
        assert meta.page_numbers == [1, 2]
        assert meta.section_title == "Introduction"
        assert meta.chunk_index == 0

    def test_empty_page_numbers(self) -> None:
        meta = ChunkMetadata(
            source_file="manual.pdf",
            page_numbers=[],
            section_title="",
            chunk_index=0,
        )
        assert meta.page_numbers == []

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            ChunkMetadata(  # type: ignore[call-arg]
                source_file="manual.pdf",
                page_numbers=[1],
                # missing section_title and chunk_index
            )


class TestDocumentChunk:
    """Test DocumentChunk model."""

    def test_valid_chunk(self) -> None:
        chunk = DocumentChunk(
            text="Some content here.",
            metadata=ChunkMetadata(
                source_file="doc.pdf",
                page_numbers=[1],
                section_title="Overview",
                chunk_index=0,
            ),
        )
        assert chunk.text == "Some content here."
        assert chunk.metadata.source_file == "doc.pdf"

    def test_empty_text_allowed(self) -> None:
        chunk = DocumentChunk(
            text="",
            metadata=ChunkMetadata(
                source_file="doc.pdf",
                page_numbers=[],
                section_title="",
                chunk_index=0,
            ),
        )
        assert chunk.text == ""


class TestParsedDocument:
    """Test ParsedDocument model."""

    def test_valid_document(self) -> None:
        doc = ParsedDocument(
            filename="manual.pdf",
            markdown="# Title\n\nContent here.",
            page_count=10,
            file_hash="abc123def456",
        )
        assert doc.filename == "manual.pdf"
        assert doc.page_count == 10
        assert doc.file_hash == "abc123def456"

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            ParsedDocument(  # type: ignore[call-arg]
                filename="manual.pdf",
                # missing markdown, page_count, file_hash
            )
