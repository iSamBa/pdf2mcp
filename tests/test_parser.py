"""Tests for pdf2mcp.parser module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.models import ParsedDocument
from pdf2mcp.parser import discover_pdfs, parse_pdf


class TestDiscoverPdfs:
    """Test PDF file discovery."""

    def test_finds_pdfs_in_directory(self, tmp_path: Path) -> None:
        (tmp_path / "doc1.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "doc2.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "readme.txt").write_text("not a pdf")

        result = discover_pdfs(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".pdf" for p in result)

    def test_finds_pdfs_in_subdirectories(self, tmp_path: Path) -> None:
        subdir = tmp_path / "nested"
        subdir.mkdir()
        (tmp_path / "top.pdf").write_bytes(b"%PDF-1.4 fake")
        (subdir / "nested.pdf").write_bytes(b"%PDF-1.4 fake")

        result = discover_pdfs(tmp_path)
        assert len(result) == 2

    def test_returns_sorted_list(self, tmp_path: Path) -> None:
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")

        result = discover_pdfs(tmp_path)
        assert result[0].name == "a.pdf"
        assert result[1].name == "b.pdf"

    def test_returns_empty_for_missing_directory(self, tmp_path: Path) -> None:
        result = discover_pdfs(tmp_path / "nonexistent")
        assert result == []

    def test_returns_empty_for_no_pdfs(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("not a pdf")
        result = discover_pdfs(tmp_path)
        assert result == []


class TestParsePdf:
    """Test PDF parsing with mocked dependencies."""

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_returns_parsed_document(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_content = b"%PDF-1.4 fake content"
        pdf_path.write_bytes(pdf_content)

        mock_pymupdf4llm.to_markdown.return_value = "# Title\n\nContent"
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 5
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_pymupdf.open.return_value = mock_doc

        result = parse_pdf(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.filename == "test.pdf"
        assert result.markdown == "# Title\n\nContent"
        assert result.page_count == 5
        assert len(result.file_hash) == 64  # SHA-256 hex digest length

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_file_hash_is_deterministic(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"same content")

        mock_pymupdf4llm.to_markdown.return_value = ""
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 1
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_pymupdf.open.return_value = mock_doc

        result1 = parse_pdf(pdf_path)
        result2 = parse_pdf(pdf_path)
        assert result1.file_hash == result2.file_hash

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_different_content_different_hash(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_pymupdf4llm.to_markdown.return_value = ""
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda self: 1
        mock_doc.__enter__ = lambda self: self
        mock_doc.__exit__ = MagicMock(return_value=False)
        mock_pymupdf.open.return_value = mock_doc

        pdf1 = tmp_path / "a.pdf"
        pdf1.write_bytes(b"content A")
        pdf2 = tmp_path / "b.pdf"
        pdf2.write_bytes(b"content B")

        assert parse_pdf(pdf1).file_hash != parse_pdf(pdf2).file_hash

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_raises_on_parse_error(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "broken.pdf"
        pdf_path.write_bytes(b"broken")
        mock_pymupdf4llm.to_markdown.side_effect = RuntimeError("parse failed")

        with pytest.raises(RuntimeError, match="parse failed"):
            parse_pdf(pdf_path)
