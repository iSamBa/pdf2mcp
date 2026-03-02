"""Tests for pdf2mcp.ingest module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pdf2mcp.ingest import run_ingestion
from pdf2mcp.models import ParsedDocument
from pdf2mcp.store import get_db, get_ingested_files


def _make_settings(tmp_path: Path) -> MagicMock:
    """Create mock settings with temporary directories."""
    s = MagicMock()
    s.docs_dir = tmp_path / "docs"
    s.docs_dir.mkdir()
    s.data_dir = tmp_path / "data"
    s.data_dir.mkdir()
    s.openai_api_key.get_secret_value.return_value = "sk-test"
    s.embedding_batch_size = 2048
    s.embedding_model = "text-embedding-3-small"
    s.embedding_dimensions = 8
    s.chunk_size = 500
    s.chunk_overlap = 50
    return s


def _make_parsed(
    filename: str = "test.pdf",
    file_hash: str = "hash123",
    markdown: str = "## Test\n\nContent here.",
) -> ParsedDocument:
    return ParsedDocument(
        filename=filename,
        markdown=markdown,
        page_count=1,
        file_hash=file_hash,
    )


class TestRunIngestion:
    """Test the ingestion orchestrator."""

    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_ingests_new_pdf(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_discover.return_value = [pdf_path]
        mock_parse.return_value = _make_parsed()
        mock_embed.return_value = [[0.1] * 8]  # One embedding for one chunk

        run_ingestion(settings=settings)

        mock_embed.assert_called_once()

    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_skips_unchanged_file(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_discover.return_value = [pdf_path]
        mock_parse.return_value = _make_parsed(file_hash="hash_v1")
        mock_embed.return_value = [[0.1] * 8]

        # First ingestion
        run_ingestion(settings=settings)
        assert mock_embed.call_count == 1

        # Second ingestion — should skip
        run_ingestion(settings=settings)
        assert mock_embed.call_count == 1  # No new embedding call

    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_reingests_changed_file(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_discover.return_value = [pdf_path]
        mock_embed.return_value = [[0.1] * 8]

        # First ingestion
        mock_parse.return_value = _make_parsed(file_hash="hash_v1")
        run_ingestion(settings=settings)
        assert mock_embed.call_count == 1

        # Second ingestion with changed hash
        mock_parse.return_value = _make_parsed(file_hash="hash_v2")
        run_ingestion(settings=settings)
        assert mock_embed.call_count == 2  # Re-embedded

    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_handles_empty_docs_dir(
        self,
        mock_discover: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        mock_discover.return_value = []

        # Should not raise
        run_ingestion(settings=settings)

    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_handles_parse_failure(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "broken.pdf"
        pdf_path.write_bytes(b"broken")

        mock_discover.return_value = [pdf_path]
        mock_parse.side_effect = RuntimeError("parse failed")

        # Should not raise
        run_ingestion(settings=settings)
        mock_embed.assert_not_called()

    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_handles_embedding_failure(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_discover.return_value = [pdf_path]
        mock_parse.return_value = _make_parsed()
        mock_embed.side_effect = RuntimeError("API error")

        # Should not raise — embedding failure is caught
        run_ingestion(settings=settings)

    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_force_reingests_all(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_discover.return_value = [pdf_path]
        mock_parse.return_value = _make_parsed(file_hash="hash_v1")
        mock_embed.return_value = [[0.1] * 8]

        # First ingestion
        run_ingestion(settings=settings)
        assert mock_embed.call_count == 1

        # Force re-ingestion — same hash but force=True
        run_ingestion(settings=settings, force=True)
        assert mock_embed.call_count == 2

    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_handles_empty_chunks(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "empty.pdf"
        pdf_path.write_bytes(b"fake")

        mock_discover.return_value = [pdf_path]
        mock_parse.return_value = _make_parsed(markdown="")  # Empty content

        # Should not raise, and should not call embed
        run_ingestion(settings=settings)
        mock_embed.assert_not_called()

    @patch("pdf2mcp.ingest.upsert_chunks")
    @patch("pdf2mcp.ingest.embed_texts")
    @patch("pdf2mcp.ingest.parse_pdf")
    @patch("pdf2mcp.ingest.discover_pdfs")
    def test_handles_store_failure(
        self,
        mock_discover: MagicMock,
        mock_parse: MagicMock,
        mock_embed: MagicMock,
        mock_upsert: MagicMock,
        tmp_path: Path,
    ) -> None:
        settings = _make_settings(tmp_path)
        pdf_path = settings.docs_dir / "test.pdf"
        pdf_path.write_bytes(b"fake")

        mock_discover.return_value = [pdf_path]
        mock_parse.return_value = _make_parsed()
        mock_embed.return_value = [[0.1] * 8]
        mock_upsert.side_effect = RuntimeError("disk full")

        # Should not raise
        run_ingestion(settings=settings)

        # File must NOT be recorded as ingested
        db = get_db(settings)
        assert get_ingested_files(db) == {}
