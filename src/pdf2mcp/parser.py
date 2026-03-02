"""PDF to Markdown extraction using pymupdf4llm."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pymupdf
import pymupdf4llm  # type: ignore[import-untyped]

from pdf2mcp.models import ParsedDocument

__all__ = ["discover_pdfs", "parse_pdf"]

logger = logging.getLogger(__name__)


def discover_pdfs(docs_dir: Path) -> list[Path]:
    """Find all PDF files recursively in the docs directory."""
    if not docs_dir.exists():
        logger.warning("Docs directory not found: %s", docs_dir)
        return []
    pdfs = sorted(docs_dir.glob("**/*.pdf"))
    logger.info("Found %d PDF files in %s", len(pdfs), docs_dir)
    return pdfs


def parse_pdf(pdf_path: Path) -> ParsedDocument:
    """Parse a single PDF file into Markdown.

    Uses pymupdf4llm for Markdown conversion and pymupdf for page count.
    Computes a SHA-256 hash of the file for change detection.
    """
    logger.info("Parsing: %s", pdf_path.name)

    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    file_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    with pymupdf.open(str(pdf_path)) as doc:  # type: ignore[no-untyped-call]
        page_count: int = len(doc)

    return ParsedDocument(
        filename=pdf_path.name,
        markdown=md_text,
        page_count=page_count,
        file_hash=file_hash,
    )
