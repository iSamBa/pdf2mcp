"""PDF to Markdown extraction using pymupdf4llm with OCR fallback."""

from __future__ import annotations

import functools
import hashlib
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pymupdf4llm  # type: ignore[import-untyped]

from pdf2mcp.models import ParsedDocument

__all__ = ["discover_pdfs", "parse_pdf"]

logger = logging.getLogger(__name__)

_MIN_TEXT_LENGTH = 10
_PAGE_BREAK = "\n-----\n\n"


def _page_has_text(page: pymupdf.Page, min_length: int = _MIN_TEXT_LENGTH) -> bool:  # type: ignore[valid-type]
    """Check whether a PDF page has extractable text content."""
    text = page.get_text().strip()  # type: ignore[attr-defined]
    return len(text) >= min_length


@functools.lru_cache(maxsize=1)
def _check_tesseract() -> bool:
    """Check if Tesseract OCR is available on the system."""
    return shutil.which("tesseract") is not None


def _ocr_page(page: pymupdf.Page, language: str = "eng", dpi: int = 300) -> str:  # type: ignore[valid-type]
    """Extract text from an image-only page using OCR.

    Returns empty string if OCR fails for any reason.
    """
    try:
        tp = page.get_textpage_ocr(language=language, dpi=dpi, full=True)  # type: ignore[attr-defined]
        text: str = page.get_text("text", textpage=tp)  # type: ignore[attr-defined]
    except Exception:
        logger.warning(
            "OCR failed for page, skipping",
            exc_info=True,
        )
        return ""
    return text.strip()


def discover_pdfs(docs_dir: Path) -> list[Path]:
    """Find all PDF files recursively in the docs directory."""
    if not docs_dir.exists():
        logger.warning("Docs directory not found: %s", docs_dir)
        return []
    pdfs = sorted(docs_dir.glob("**/*.pdf"))
    logger.info("Found %d PDF files in %s", len(pdfs), docs_dir)
    return pdfs


def parse_pdf(
    pdf_path: Path,
    *,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
    ocr_dpi: int = 300,
    on_ocr_start: Callable[[int], None] | None = None,
    on_ocr_page: Callable[[], None] | None = None,
) -> ParsedDocument:
    """Parse a single PDF file into Markdown.

    Uses a per-page strategy: text pages are extracted via pymupdf4llm,
    image-only pages are OCR'd via Tesseract (if available and enabled).
    Computes a SHA-256 hash of the file for change detection.
    """
    logger.info("Parsing: %s", pdf_path.name)

    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    with pymupdf.open(str(pdf_path)) as doc:  # type: ignore[no-untyped-call]
        page_count: int = len(doc)

        # Classify pages
        image_only_pages: set[int] = set()
        for i, page in enumerate(doc):
            if not _page_has_text(page):
                image_only_pages.add(i)

        ocr_page_count = len(image_only_pages)
        can_ocr = (
            ocr_enabled
            and ocr_page_count > 0
            and _check_tesseract()
        )

        if ocr_page_count > 0 and not ocr_enabled:
            logger.info(
                "OCR disabled — skipping %d image-only page(s) in %s",
                ocr_page_count,
                pdf_path.name,
            )
        elif ocr_page_count > 0 and not can_ocr:
            logger.warning(
                "Tesseract not found — skipping OCR for %d image-only "
                "page(s) in %s. Install Tesseract for scanned PDF support.",
                ocr_page_count,
                pdf_path.name,
            )

        # Build markdown page-by-page
        page_markdowns: list[str] = []
        ocr_start = time.monotonic() if can_ocr else 0.0

        if can_ocr and on_ocr_start is not None:
            on_ocr_start(ocr_page_count)

        for i in range(page_count):
            if i not in image_only_pages:
                # Text page: extract via pymupdf4llm (one page at a time)
                page_md: str = pymupdf4llm.to_markdown(
                    doc, pages=[i]
                )
                stripped = page_md.strip()
                if stripped:
                    page_markdowns.append(stripped)
            elif can_ocr:
                # Image-only page: OCR via Tesseract
                ocr_text = _ocr_page(doc[i], language=ocr_language, dpi=ocr_dpi)
                if ocr_text:
                    page_markdowns.append(ocr_text)
                if on_ocr_page is not None:
                    on_ocr_page()
            # If image-only and no tesseract, skip (already warned)

        md_text = _PAGE_BREAK.join(page_markdowns)

    if ocr_page_count > 0 and can_ocr:
        elapsed = time.monotonic() - ocr_start
        logger.info(
            "OCR completed: %d page(s) in %.1fs for %s",
            ocr_page_count,
            elapsed,
            pdf_path.name,
        )

    return ParsedDocument(
        filename=pdf_path.name,
        markdown=md_text,
        page_count=page_count,
        file_hash=file_hash,
        ocr_pages=ocr_page_count if can_ocr else 0,
    )
