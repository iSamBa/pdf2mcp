"""Tests for pdf2mcp.parser module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.models import ParsedDocument
from pdf2mcp.parser import _page_has_text, discover_pdfs, parse_pdf


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


class TestPageHasText:
    """Test the _page_has_text detection function."""

    def test_page_with_sufficient_text(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "This is a page with enough text content."
        assert _page_has_text(page) is True

    def test_page_with_empty_text(self) -> None:
        page = MagicMock()
        page.get_text.return_value = ""
        assert _page_has_text(page) is False

    def test_page_with_only_whitespace(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "   \n\t  \n  "
        assert _page_has_text(page) is False

    def test_page_with_short_text_below_threshold(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "abc"  # 3 chars < 10
        assert _page_has_text(page) is False

    def test_page_with_text_at_threshold(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "0123456789"  # exactly 10 chars
        assert _page_has_text(page) is True

    def test_page_with_text_just_below_threshold(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "012345678"  # 9 chars
        assert _page_has_text(page) is False

    def test_custom_min_length(self) -> None:
        page = MagicMock()
        page.get_text.return_value = "abc"
        assert _page_has_text(page, min_length=3) is True
        assert _page_has_text(page, min_length=4) is False


def _make_mock_page(text: str) -> MagicMock:
    """Create a mock pymupdf page with given text."""
    page = MagicMock()
    page.get_text.return_value = text
    return page


def _make_mock_doc(pages: list[MagicMock]) -> MagicMock:
    """Create a mock pymupdf document with given pages."""
    mock_doc = MagicMock()
    mock_doc.__len__ = lambda self: len(pages)
    mock_doc.__iter__ = lambda self: iter(pages)
    mock_doc.__enter__ = lambda self: self
    mock_doc.__exit__ = MagicMock(return_value=False)
    return mock_doc


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
        pages = [_make_mock_page("This page has enough text for detection.")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages * 5)

        result = parse_pdf(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.filename == "test.pdf"
        assert result.markdown == "# Title\n\nContent"
        assert result.page_count == 5
        assert len(result.file_hash) == 64
        assert result.ocr_pages == 0

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
        pages = [_make_mock_page("enough text here")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

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
        pages = [_make_mock_page("enough text here")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

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

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_text_pdf_has_zero_ocr_pages(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "text.pdf"
        pdf_path.write_bytes(b"text pdf content")

        mock_pymupdf4llm.to_markdown.return_value = "# Text document"
        pages = [
            _make_mock_page("Page one with lots of text content here."),
            _make_mock_page("Page two also has plenty of text content."),
        ]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        result = parse_pdf(pdf_path)
        assert result.ocr_pages == 0

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_all_image_only_pages(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        mock_pymupdf4llm.to_markdown.return_value = ""
        pages = [
            _make_mock_page(""),
            _make_mock_page("   "),
            _make_mock_page("ab"),
        ]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        result = parse_pdf(pdf_path)
        assert result.ocr_pages == 3
        assert result.ocr_pages == result.page_count

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_mixed_pdf_counts_image_only_pages(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "mixed.pdf"
        pdf_path.write_bytes(b"mixed pdf")

        mock_pymupdf4llm.to_markdown.return_value = "# Some content"
        pages = [
            _make_mock_page("This page has real text content here."),
            _make_mock_page(""),  # image-only
            _make_mock_page("Another page with text content in it."),
            _make_mock_page("  \n  "),  # image-only
        ]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        result = parse_pdf(pdf_path)
        assert result.page_count == 4
        assert result.ocr_pages == 2

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_logs_warning_for_image_only_pages(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        mock_pymupdf4llm.to_markdown.return_value = ""
        pages = [_make_mock_page(""), _make_mock_page(""), _make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        with caplog.at_level(logging.WARNING, logger="pdf2mcp.parser"):
            parse_pdf(pdf_path)

        assert "3 image-only page(s)" in caplog.text
        assert "scanned.pdf" in caplog.text

    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_no_warning_for_text_only_pdf(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pdf_path = tmp_path / "text.pdf"
        pdf_path.write_bytes(b"text pdf")

        mock_pymupdf4llm.to_markdown.return_value = "# Content"
        pages = [_make_mock_page("Plenty of text content on this page.")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        with caplog.at_level(logging.WARNING, logger="pdf2mcp.parser"):
            parse_pdf(pdf_path)

        assert "image-only" not in caplog.text
