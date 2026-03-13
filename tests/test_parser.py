"""Tests for pdf2mcp.parser module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.models import ParsedDocument
from pdf2mcp.parser import (
    _check_tesseract,
    _ocr_page,
    _page_has_text,
    discover_pdfs,
    parse_pdf,
)


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

    @staticmethod
    def _make_text_page(text: str) -> MagicMock:
        """Create a mock page with text but no images (pure text page)."""
        page = MagicMock()
        page.get_text.return_value = text
        page.get_images.return_value = []
        page.rect.width = 612
        page.rect.height = 792
        return page

    def test_page_with_sufficient_text(self) -> None:
        page = self._make_text_page("This is a page with enough text content.")
        assert _page_has_text(page) is True

    def test_page_with_empty_text(self) -> None:
        page = self._make_text_page("")
        assert _page_has_text(page) is False

    def test_page_with_only_whitespace(self) -> None:
        page = self._make_text_page("   \n\t  \n  ")
        assert _page_has_text(page) is False

    def test_page_with_short_text_below_threshold(self) -> None:
        page = self._make_text_page("abc")  # 3 chars < 10
        assert _page_has_text(page) is False

    def test_page_with_text_at_threshold(self) -> None:
        page = self._make_text_page("0123456789")  # exactly 10 chars
        assert _page_has_text(page) is True

    def test_page_with_text_just_below_threshold(self) -> None:
        page = self._make_text_page("012345678")  # 9 chars
        assert _page_has_text(page) is False

    def test_custom_min_length(self) -> None:
        page = self._make_text_page("abc")
        assert _page_has_text(page, min_length=3) is True
        assert _page_has_text(page, min_length=4) is False

    def test_image_dominant_page_with_text_returns_false(self) -> None:
        """A scanned page with overlay text should not be treated as a text page."""
        page = MagicMock()
        page.get_text.return_value = "SAMPLE LETTER"  # 13 chars >= 10
        page.get_images.return_value = [(1,)]  # one image
        rect = MagicMock()
        rect.width = 612
        rect.height = 792
        page.rect = rect
        page.get_image_rects.return_value = [rect]  # image covers full page
        assert _page_has_text(page) is False


class TestCheckTesseract:
    """Test Tesseract availability detection."""

    @patch("pdf2mcp.parser.shutil.which")
    def test_returns_true_when_tesseract_found(self, mock_which: MagicMock) -> None:
        _check_tesseract.cache_clear()
        mock_which.return_value = "/usr/local/bin/tesseract"
        assert _check_tesseract() is True
        mock_which.assert_called_once_with("tesseract")

    @patch("pdf2mcp.parser.shutil.which")
    def test_returns_false_when_tesseract_missing(self, mock_which: MagicMock) -> None:
        _check_tesseract.cache_clear()
        mock_which.return_value = None
        assert _check_tesseract() is False


class TestOcrPage:
    """Test OCR text extraction from a page."""

    def test_returns_ocr_text(self) -> None:
        page = MagicMock()
        mock_tp = MagicMock()
        page.get_textpage_ocr.return_value = mock_tp
        page.get_text.return_value = "  OCR extracted text content  "

        result = _ocr_page(page)

        assert result == "OCR extracted text content"
        page.get_textpage_ocr.assert_called_once_with(
            language="eng", dpi=300, full=True
        )
        page.get_text.assert_called_once_with("text", textpage=mock_tp)

    def test_returns_empty_for_blank_page(self) -> None:
        page = MagicMock()
        page.get_textpage_ocr.return_value = MagicMock()
        page.get_text.return_value = "   "

        result = _ocr_page(page)
        assert result == ""

    def test_custom_language(self) -> None:
        page = MagicMock()
        page.get_textpage_ocr.return_value = MagicMock()
        page.get_text.return_value = "Texte français"

        _ocr_page(page, language="fra")

        page.get_textpage_ocr.assert_called_once_with(
            language="fra", dpi=300, full=True
        )

    def test_near_threshold_ocr_text(self) -> None:
        page = MagicMock()
        page.get_textpage_ocr.return_value = MagicMock()
        page.get_text.return_value = "short"

        result = _ocr_page(page)
        assert result == "short"

    def test_returns_empty_on_ocr_failure(self) -> None:
        page = MagicMock()
        page.get_textpage_ocr.side_effect = RuntimeError("Tesseract crashed")

        result = _ocr_page(page)
        assert result == ""

    def test_logs_warning_on_ocr_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        page = MagicMock()
        page.get_textpage_ocr.side_effect = RuntimeError("corrupt image")

        with caplog.at_level(logging.WARNING, logger="pdf2mcp.parser"):
            _ocr_page(page)

        assert "OCR failed for page" in caplog.text


def _make_mock_page(text: str, *, image_dominant: bool = False) -> MagicMock:
    """Create a mock pymupdf page with given text."""
    page = MagicMock()
    page.get_text.return_value = text
    page.rect.width = 612
    page.rect.height = 792
    if image_dominant:
        page.get_images.return_value = [(1,)]
        full_rect = MagicMock()
        full_rect.width = 612
        full_rect.height = 792
        page.get_image_rects.return_value = [full_rect]
    else:
        page.get_images.return_value = []
    return page


def _make_mock_doc(pages: list[MagicMock]) -> MagicMock:
    """Create a mock pymupdf document with given pages."""
    mock_doc = MagicMock()
    mock_doc.__len__ = lambda self: len(pages)
    mock_doc.__iter__ = lambda self: iter(pages)
    mock_doc.__enter__ = lambda self: self
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_doc.__getitem__ = lambda self, idx: pages[idx]
    return mock_doc


class TestParsePdf:
    """Test PDF parsing with mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _clear_tesseract_cache(self) -> None:
        _check_tesseract.cache_clear()

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_returns_parsed_document(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")

        mock_pymupdf4llm.to_markdown.return_value = "# Title\n\nContent"
        pages = [_make_mock_page("This page has enough text.") for _ in range(5)]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        result = parse_pdf(pdf_path)

        assert isinstance(result, ParsedDocument)
        assert result.filename == "test.pdf"
        assert result.page_count == 5
        assert len(result.file_hash) == 64
        assert result.ocr_pages == 0

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_file_hash_is_deterministic(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"same content")

        mock_pymupdf4llm.to_markdown.return_value = "text"
        pages = [_make_mock_page("enough text here")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        result1 = parse_pdf(pdf_path)
        result2 = parse_pdf(pdf_path)
        assert result1.file_hash == result2.file_hash

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_different_content_different_hash(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_pymupdf4llm.to_markdown.return_value = "text"
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

        pages = [_make_mock_page("enough text here")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_pymupdf4llm.to_markdown.side_effect = RuntimeError("parse failed")

        with pytest.raises(RuntimeError, match="parse failed"):
            parse_pdf(pdf_path)

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_text_pdf_has_zero_ocr_pages(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
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

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_all_image_only_pages_no_tesseract(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [
            _make_mock_page(""),
            _make_mock_page("   "),
            _make_mock_page("ab"),
        ]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        result = parse_pdf(pdf_path)
        assert result.ocr_pages == 0  # Tesseract missing → no OCR performed
        assert result.markdown == ""
        mock_pymupdf4llm.to_markdown.assert_not_called()

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_all_image_only_pages_with_tesseract(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page(""), _make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.side_effect = ["OCR text page 1", "OCR text page 2"]

        result = parse_pdf(pdf_path)
        assert result.ocr_pages == 2
        assert "OCR text page 1" in result.markdown
        assert "OCR text page 2" in result.markdown
        assert mock_ocr_page.call_count == 2

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_mixed_pdf_combines_text_and_ocr(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "mixed.pdf"
        pdf_path.write_bytes(b"mixed pdf")

        pages = [
            _make_mock_page("This page has real text content here."),
            _make_mock_page(""),  # image-only
            _make_mock_page("Another page with text content in it."),
        ]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        # Per-page calls: page 0 and page 2 are text pages
        mock_pymupdf4llm.to_markdown.side_effect = [
            "Text page 0 content",
            "Text page 2 content",
        ]
        mock_ocr_page.return_value = "OCR scanned content"

        result = parse_pdf(pdf_path)
        assert result.page_count == 3
        assert result.ocr_pages == 1
        assert "Text page 0 content" in result.markdown
        assert "OCR scanned content" in result.markdown
        assert "Text page 2 content" in result.markdown
        # Verify page break markers exist between pages
        assert "\n-----\n" in result.markdown
        # pymupdf4llm called once per text page
        assert mock_pymupdf4llm.to_markdown.call_count == 2

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_logs_warning_when_tesseract_missing(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page(""), _make_mock_page(""), _make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        with caplog.at_level(logging.WARNING, logger="pdf2mcp.parser"):
            parse_pdf(pdf_path)

        assert "Tesseract not found" in caplog.text
        assert "3 image-only page(s)" in caplog.text
        assert "scanned.pdf" in caplog.text

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_no_warning_for_text_only_pdf(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
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

        assert "Tesseract" not in caplog.text
        assert "image-only" not in caplog.text

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_graceful_degradation_mixed_pdf_no_tesseract(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Text pages still extracted when Tesseract is missing."""
        pdf_path = tmp_path / "mixed.pdf"
        pdf_path.write_bytes(b"mixed pdf")

        pages = [
            _make_mock_page("This page has real text content here."),
            _make_mock_page(""),  # image-only — will be skipped
        ]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_pymupdf4llm.to_markdown.return_value = "Text page content"

        result = parse_pdf(pdf_path)
        assert result.ocr_pages == 0  # Tesseract missing → no OCR performed
        assert "Text page content" in result.markdown

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_ocr_empty_result_excluded(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """OCR page that produces empty text is excluded from markdown."""
        pdf_path = tmp_path / "blank.pdf"
        pdf_path.write_bytes(b"blank pdf")

        pages = [_make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.return_value = ""

        result = parse_pdf(pdf_path)
        assert result.markdown == ""

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_logs_info_after_ocr(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.return_value = "OCR text"

        with caplog.at_level(logging.INFO, logger="pdf2mcp.parser"):
            parse_pdf(pdf_path)

        assert "OCR completed: 1 page(s)" in caplog.text

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_page_breaks_between_pages(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "multipage.pdf"
        pdf_path.write_bytes(b"multipage pdf")

        pages = [
            _make_mock_page("Page 1 has enough text content here."),
            _make_mock_page("Page 2 has enough text content here."),
        ]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_pymupdf4llm.to_markdown.side_effect = ["Page 1 md", "Page 2 md"]

        result = parse_pdf(pdf_path)
        assert "\n-----\n" in result.markdown
        assert "Page 1 md" in result.markdown
        assert "Page 2 md" in result.markdown

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_text_with_dashes_not_misinterpreted(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Content containing ----- is not misinterpreted as page break."""
        pdf_path = tmp_path / "dashes.pdf"
        pdf_path.write_bytes(b"dashes pdf")

        pages = [_make_mock_page("Text with ----- in it and more text.")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_pymupdf4llm.to_markdown.return_value = "Content with -----\nrule in middle"

        result = parse_pdf(pdf_path)
        assert "-----" in result.markdown
        assert "Content with" in result.markdown
        assert "rule in middle" in result.markdown

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_ocr_disabled_skips_all_ocr(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When ocr_enabled=False, no OCR is performed even if Tesseract exists."""
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page(""), _make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        result = parse_pdf(pdf_path, ocr_enabled=False)
        assert result.ocr_pages == 0  # OCR disabled → no OCR performed
        assert result.markdown == ""
        mock_ocr_page.assert_not_called()

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_ocr_language_passed_to_ocr_page(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "german.pdf"
        pdf_path.write_bytes(b"german pdf")

        pages = [_make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.return_value = "Deutscher Text"

        parse_pdf(pdf_path, ocr_language="deu")
        mock_ocr_page.assert_called_once_with(pages[0], language="deu", dpi=300)

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_ocr_dpi_passed_to_ocr_page(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "lowdpi.pdf"
        pdf_path.write_bytes(b"lowdpi pdf")

        pages = [_make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.return_value = "OCR text"

        parse_pdf(pdf_path, ocr_dpi=150)
        mock_ocr_page.assert_called_once_with(pages[0], language="eng", dpi=150)

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_ocr_disabled_logs_info(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        with caplog.at_level(logging.INFO, logger="pdf2mcp.parser"):
            parse_pdf(pdf_path, ocr_enabled=False)

        assert "OCR disabled" in caplog.text

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_on_ocr_page_callback_called_per_page(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page(""), _make_mock_page(""), _make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.return_value = "OCR text"

        callback = MagicMock()
        parse_pdf(pdf_path, on_ocr_page=callback)
        assert callback.call_count == 3

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_on_ocr_page_callback_not_called_for_text_pdf(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "text.pdf"
        pdf_path.write_bytes(b"text pdf")

        mock_pymupdf4llm.to_markdown.return_value = "# Content"
        pages = [_make_mock_page("Plenty of text content on this page.")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        callback = MagicMock()
        parse_pdf(pdf_path, on_ocr_page=callback)
        callback.assert_not_called()

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_on_ocr_start_callback_called_with_total(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page(""), _make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.return_value = "OCR text"

        start_callback = MagicMock()
        parse_pdf(pdf_path, on_ocr_start=start_callback)
        start_callback.assert_called_once_with(2)

    @patch("pdf2mcp.parser._check_tesseract", return_value=False)
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_on_ocr_start_not_called_when_no_ocr(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
    ) -> None:
        pdf_path = tmp_path / "text.pdf"
        pdf_path.write_bytes(b"text pdf")

        mock_pymupdf4llm.to_markdown.return_value = "# Content"
        pages = [_make_mock_page("Plenty of text content on this page.")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)

        start_callback = MagicMock()
        parse_pdf(pdf_path, on_ocr_start=start_callback)
        start_callback.assert_not_called()

    @patch("pdf2mcp.parser._check_tesseract", return_value=True)
    @patch("pdf2mcp.parser._ocr_page")
    @patch("pdf2mcp.parser.pymupdf")
    @patch("pdf2mcp.parser.pymupdf4llm")
    def test_ocr_timing_in_log(
        self,
        mock_pymupdf4llm: MagicMock,
        mock_pymupdf: MagicMock,
        mock_ocr_page: MagicMock,
        _mock_tess: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(b"scanned pdf")

        pages = [_make_mock_page("")]
        mock_pymupdf.open.return_value = _make_mock_doc(pages)
        mock_ocr_page.return_value = "OCR text"

        with caplog.at_level(logging.INFO, logger="pdf2mcp.parser"):
            parse_pdf(pdf_path)

        # Verify timing is included in log
        assert "OCR completed: 1 page(s) in" in caplog.text
        assert "s for scanned.pdf" in caplog.text
