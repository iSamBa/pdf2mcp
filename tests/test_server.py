"""Tests for pdf2mcp.server module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pdf2mcp.search import SearchResult
from pdf2mcp.server import (
    get_sections,
    get_status,
    list_docs,
    read_page,
    read_section,
    search_docs,
    search_in_doc,
)


def _make_settings() -> MagicMock:
    """Create a mock Settings object."""
    s = MagicMock()
    s.embedding_model = "text-embedding-3-small"
    s.docs_dir = "docs"
    return s


# ── search_docs ────────────────────────────────────────────────────


class TestSearchDocs:
    """Test the search_docs tool."""

    @patch("pdf2mcp.server.search_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_returns_formatted_results(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = [
            SearchResult(
                text="PDF document specs",
                score=0.1,
                source_file="manual.pdf",
                page_numbers=[5],
                section_title="Specs",
            )
        ]

        result = search_docs("document specs")
        assert "PDF document specs" in result
        assert "manual.pdf" in result
        mock_search.assert_called_once()

    @patch("pdf2mcp.server.search_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_empty_query_returns_no_results_message(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = []

        result = search_docs("")
        assert "No results found" in result

    @patch("pdf2mcp.server.search_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_clamps_num_results_low(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = []

        search_docs("query", num_results=-5)
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["num_results"] == 1

    @patch("pdf2mcp.server.search_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_clamps_num_results_high(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = []

        search_docs("query", num_results=100)
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["num_results"] == 20

    @patch("pdf2mcp.server.search_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_handles_search_failure(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.side_effect = RuntimeError("API error")

        result = search_docs("query")
        assert "Search failed" in result

    @patch("pdf2mcp.server.search_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_default_num_results(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = []

        search_docs("query")
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["num_results"] == 5

    @patch("pdf2mcp.server.get_settings")
    def test_handles_settings_failure(self, mock_settings: MagicMock) -> None:
        mock_settings.side_effect = ValueError("OPENAI_API_KEY is required")

        result = search_docs("query")
        assert "Search failed" in result


# ── list_docs ──────────────────────────────────────────────────────


class TestListDocs:
    """Test the list_docs tool."""

    @patch("pdf2mcp.server.list_ingested_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_returns_document_list(
        self, mock_settings: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_list.return_value = [
            {"filename": "manual.pdf", "file_hash": "abc12345def", "chunk_count": 42},
            {"filename": "guide.pdf", "file_hash": "xyz98765uvw", "chunk_count": 18},
        ]

        result = list_docs()
        assert "manual.pdf" in result
        assert "guide.pdf" in result
        assert "42 chunks" in result
        assert "Total: 2 documents" in result

    @patch("pdf2mcp.server.list_ingested_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_no_documents_message(
        self, mock_settings: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_list.return_value = []

        result = list_docs()
        assert "No documents ingested" in result
        assert "pdf2mcp ingest" in result

    @patch("pdf2mcp.server.list_ingested_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_handles_failure(
        self, mock_settings: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_list.side_effect = RuntimeError("disk error")

        result = list_docs()
        assert "Failed" in result

    @patch("pdf2mcp.server.list_ingested_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_truncates_file_hash(
        self, mock_settings: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_list.return_value = [
            {"filename": "test.pdf", "file_hash": "abcdef1234567890", "chunk_count": 5},
        ]

        result = list_docs()
        assert "abcdef12..." in result
        # Full hash should not appear
        assert "abcdef1234567890" not in result

    @patch("pdf2mcp.server.get_settings")
    def test_handles_settings_failure(self, mock_settings: MagicMock) -> None:
        mock_settings.side_effect = ValueError("Invalid config")

        result = list_docs()
        assert "Failed" in result


# ── get_sections ───────────────────────────────────────────────────


class TestGetSections:
    """Test the get_sections tool."""

    @patch("pdf2mcp.server.get_document_sections")
    @patch("pdf2mcp.server.get_settings")
    def test_returns_sections(
        self, mock_settings: MagicMock, mock_sections: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_sections.return_value = ["Introduction", "Safety", "Programming"]

        result = get_sections("manual.pdf")
        assert "Sections in manual.pdf" in result
        assert "1. Introduction" in result
        assert "2. Safety" in result
        assert "3. Programming" in result

    @patch("pdf2mcp.server.get_document_sections")
    @patch("pdf2mcp.server.get_settings")
    def test_unknown_filename(
        self, mock_settings: MagicMock, mock_sections: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_sections.return_value = []

        result = get_sections("unknown.pdf")
        assert "No sections found" in result
        assert "unknown.pdf" in result

    @patch("pdf2mcp.server.get_document_sections")
    @patch("pdf2mcp.server.get_settings")
    def test_handles_failure(
        self, mock_settings: MagicMock, mock_sections: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_sections.side_effect = RuntimeError("DB error")

        result = get_sections("test.pdf")
        assert "Failed" in result

    @patch("pdf2mcp.server.get_settings")
    def test_handles_settings_failure(self, mock_settings: MagicMock) -> None:
        mock_settings.side_effect = ValueError("Invalid config")

        result = get_sections("test.pdf")
        assert "Failed" in result


# ── get_status ─────────────────────────────────────────────────────


class TestGetStatus:
    """Test the get_status resource."""

    @patch("pdf2mcp.server.list_ingested_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_returns_status(
        self, mock_settings: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_list.return_value = [
            {"filename": "a.pdf", "file_hash": "h1", "chunk_count": 10},
            {"filename": "b.pdf", "file_hash": "h2", "chunk_count": 20},
        ]

        result = get_status()
        assert "Documents: 2" in result
        assert "Total chunks: 30" in result
        assert "text-embedding-3-small" in result

    @patch("pdf2mcp.server.list_ingested_documents")
    @patch("pdf2mcp.server.get_settings")
    def test_empty_database(
        self, mock_settings: MagicMock, mock_list: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_list.return_value = []

        result = get_status()
        assert "Documents: 0" in result
        assert "Total chunks: 0" in result

    @patch("pdf2mcp.server.get_settings")
    def test_handles_failure(self, mock_settings: MagicMock) -> None:
        mock_settings.side_effect = RuntimeError("config error")

        result = get_status()
        assert "no documents ingested" in result


# ── search_in_doc ─────────────────────────────────────────────────


class TestSearchInDoc:
    """Test the search_in_doc tool."""

    @patch("pdf2mcp.server.search_in_document")
    @patch("pdf2mcp.server.get_settings")
    def test_returns_formatted_results(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = [
            SearchResult(
                text="Torque specs",
                score=0.1,
                source_file="manual.pdf",
                page_numbers=[5],
                section_title="Specs",
            )
        ]

        result = search_in_doc("torque", "manual.pdf")
        assert "Torque specs" in result
        assert "manual.pdf" in result
        mock_search.assert_called_once()

    @patch("pdf2mcp.server.search_in_document")
    @patch("pdf2mcp.server.get_settings")
    def test_clamps_num_results(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = []

        search_in_doc("query", "test.pdf", num_results=100)
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["num_results"] == 20

    @patch("pdf2mcp.server.search_in_document")
    @patch("pdf2mcp.server.get_settings")
    def test_empty_results(
        self, mock_settings: MagicMock, mock_search: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_search.return_value = []

        result = search_in_doc("query", "test.pdf")
        assert "No results found" in result

    @patch("pdf2mcp.server.get_settings")
    def test_handles_failure(self, mock_settings: MagicMock) -> None:
        mock_settings.side_effect = RuntimeError("error")

        result = search_in_doc("query", "test.pdf")
        assert "failed" in result.lower()


# ── read_page ─────────────────────────────────────────────────────


class TestReadPage:
    """Test the read_page tool."""

    @patch("pdf2mcp.server.get_page_chunks")
    @patch("pdf2mcp.server.get_settings")
    def test_returns_formatted_page(
        self, mock_settings: MagicMock, mock_chunks: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_chunks.return_value = [
            {
                "text": "Page content here",
                "source_file": "manual.pdf",
                "page_numbers": [5],
                "section_title": "Intro",
                "chunk_index": 0,
            }
        ]

        result = read_page("manual.pdf", 5)
        assert "Page 5" in result
        assert "manual.pdf" in result
        assert "Page content here" in result

    @patch("pdf2mcp.server.get_page_chunks")
    @patch("pdf2mcp.server.get_settings")
    def test_page_not_found(
        self, mock_settings: MagicMock, mock_chunks: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_chunks.return_value = []

        result = read_page("manual.pdf", 999)
        assert "No content found" in result

    @patch("pdf2mcp.server.get_settings")
    def test_handles_failure(self, mock_settings: MagicMock) -> None:
        mock_settings.side_effect = RuntimeError("error")

        result = read_page("test.pdf", 1)
        assert "Failed" in result


# ── read_section ──────────────────────────────────────────────────


class TestReadSection:
    """Test the read_section tool."""

    @patch("pdf2mcp.server.get_section_chunks")
    @patch("pdf2mcp.server.get_settings")
    def test_returns_formatted_section(
        self, mock_settings: MagicMock, mock_chunks: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_chunks.return_value = [
            {
                "text": "Safety content",
                "source_file": "manual.pdf",
                "page_numbers": [3, 4],
                "section_title": "Safety",
                "chunk_index": 0,
            }
        ]

        result = read_section("manual.pdf", "Safety")
        assert "Safety" in result
        assert "manual.pdf" in result
        assert "Safety content" in result

    @patch("pdf2mcp.server.get_section_chunks")
    @patch("pdf2mcp.server.get_settings")
    def test_section_not_found(
        self, mock_settings: MagicMock, mock_chunks: MagicMock
    ) -> None:
        mock_settings.return_value = _make_settings()
        mock_chunks.return_value = []

        result = read_section("manual.pdf", "Nonexistent")
        assert "No content found" in result

    @patch("pdf2mcp.server.get_settings")
    def test_handles_failure(self, mock_settings: MagicMock) -> None:
        mock_settings.side_effect = RuntimeError("error")

        result = read_section("test.pdf", "Intro")
        assert "Failed" in result
