"""E2E tests for CLI commands.

Uses in-process CLI function calls with real LanceDB, real PDF
parsing, and mocked embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pdf2mcp.cli import cmd_config, cmd_delete, cmd_ingest, cmd_search, cmd_stats

pytestmark = pytest.mark.e2e


def _make_args(**kwargs: object) -> argparse.Namespace:
    """Build a minimal argparse.Namespace with defaults."""
    defaults = {
        "verbose": False,
        "force": False,
        "docs_dir": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestInit:
    def test_init_creates_scaffold(self, tmp_path: Path) -> None:
        from pdf2mcp.cli import cmd_init

        args = _make_args(directory=str(tmp_path / "myproject"), interactive=False)
        cmd_init(args)

        project = tmp_path / "myproject"
        assert (project / "docs").is_dir()
        assert (project / ".env").is_file()


class TestIngest:
    def test_ingest_processes_pdf(self, e2e_project: Path) -> None:
        args = _make_args()
        cmd_ingest(args)

        from pdf2mcp.config import get_settings
        from pdf2mcp.search import list_ingested_documents

        settings = get_settings()
        docs = list_ingested_documents(settings)
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.pdf"
        assert docs[0]["chunk_count"] > 0

    def test_ingest_force_reingests(self, e2e_project: Path) -> None:
        # First ingest
        cmd_ingest(_make_args())

        from pdf2mcp.config import get_settings
        from pdf2mcp.search import list_ingested_documents

        settings = get_settings()
        first_count = list_ingested_documents(settings)[0]["chunk_count"]

        # Clear caches so second ingest sees fresh state
        from pdf2mcp.store import _cached_connect, invalidate_table_cache

        get_settings.cache_clear()
        _cached_connect.cache_clear()
        invalidate_table_cache()

        # Force re-ingest
        cmd_ingest(_make_args(force=True))

        get_settings.cache_clear()
        _cached_connect.cache_clear()
        invalidate_table_cache()

        settings = get_settings()
        second_count = list_ingested_documents(settings)[0]["chunk_count"]
        assert second_count == first_count


class TestStats:
    def test_stats_shows_documents(
        self, ingested_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = _make_args()
        cmd_stats(args)

        captured = capsys.readouterr()
        # Rich output goes to stderr
        assert "test.pdf" in captured.err or "Documents" in captured.err


class TestSearch:
    def test_search_returns_results(
        self, ingested_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = _make_args(query="safety", num_results=5, filename=None)
        cmd_search(args)

        captured = capsys.readouterr()
        assert "Result 1" in captured.err

    def test_search_with_filename_filter(
        self, ingested_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = _make_args(query="safety", num_results=5, filename="test.pdf")
        cmd_search(args)

        captured = capsys.readouterr()
        assert "Result 1" in captured.err


class TestDelete:
    def test_delete_removes_document(self, ingested_project: Path) -> None:
        args = _make_args(filename="test.pdf", yes=True)
        cmd_delete(args)

        from pdf2mcp.config import get_settings
        from pdf2mcp.search import list_ingested_documents
        from pdf2mcp.store import _cached_connect, invalidate_table_cache

        get_settings.cache_clear()
        _cached_connect.cache_clear()
        invalidate_table_cache()

        settings = get_settings()
        docs = list_ingested_documents(settings)
        assert len(docs) == 0


class TestConfig:
    def test_config_outputs_valid_json(
        self, e2e_project: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = _make_args(
            name=None,
            transport=None,
            url=None,
            client="claude-code",
        )
        cmd_config(args)

        captured = capsys.readouterr()
        # Extract JSON from stdout (skips comment lines)
        json_lines = [
            line
            for line in captured.out.splitlines()
            if not line.startswith("#") and line.strip()
        ]
        parsed = json.loads("\n".join(json_lines))
        assert "mcpServers" in parsed
