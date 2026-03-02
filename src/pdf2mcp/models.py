"""Data models for the PDF parsing and chunking pipeline."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["ChunkMetadata", "DocumentChunk", "ParsedDocument"]


class ChunkMetadata(BaseModel):
    """Metadata attached to each text chunk."""

    source_file: str
    page_numbers: list[int]
    section_title: str
    chunk_index: int


class DocumentChunk(BaseModel):
    """A single chunk of text with its metadata."""

    text: str
    metadata: ChunkMetadata


class ParsedDocument(BaseModel):
    """Result of parsing a single PDF file."""

    filename: str
    markdown: str
    page_count: int
    file_hash: str
