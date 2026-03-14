"""Tests for pdf2mcp.cli module."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.cli import (
    cmd_config,
    cmd_delete,
    cmd_ingest,
    cmd_init,
    cmd_search,
    cmd_serve,
    cmd_stats,
    main,
    setup_logging,
)


def _extract_json(output: str) -> dict:  # type: ignore[type-arg]
    """Extract the first JSON object from config output (skips comment lines)."""
    lines = output.strip().splitlines()
    json_lines = [line for line in lines if not line.startswith("#")]
    return json.loads("\n".join(json_lines))  # type: ignore[no-any-return]


# ── setup_logging ─────────────────────────────────────────────────


class TestSetupLogging:
    """Test logging configuration."""

    def test_configures_stderr_handler(self) -> None:
        # Reset logging to test fresh config
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        setup_logging(verbose=False)

        root = logging.getLogger()
        assert root.level == logging.INFO
        assert any(getattr(h, "stream", None) is sys.stderr for h in root.handlers)

    def test_verbose_sets_debug_level(self) -> None:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)

        setup_logging(verbose=True)

        root = logging.getLogger()
        assert root.level == logging.DEBUG


# ── main() ────────────────────────────────────────────────────────


class TestMain:
    """Test main CLI entry point."""

    def test_no_args_prints_help_and_exits(self) -> None:
        with patch("sys.argv", ["pdf2mcp"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        with patch("sys.argv", ["pdf2mcp", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        from pdf2mcp import __version__

        assert __version__ in captured.out

    @patch("pdf2mcp.cli.cmd_ingest")
    def test_ingest_subcommand_calls_cmd_ingest(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "ingest"]):
            main()
        mock_cmd.assert_called_once()

    @patch("pdf2mcp.cli.cmd_serve")
    def test_serve_subcommand_calls_cmd_serve(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "serve"]):
            main()
        mock_cmd.assert_called_once()

    @patch("pdf2mcp.cli.cmd_config")
    def test_config_subcommand_calls_cmd_config(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "config"]):
            main()
        mock_cmd.assert_called_once()

    @patch("pdf2mcp.cli.cmd_init")
    def test_init_subcommand_calls_cmd_init(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "init"]):
            main()
        mock_cmd.assert_called_once()

    @patch("pdf2mcp.cli.cmd_delete")
    def test_delete_subcommand_calls_cmd_delete(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "delete", "test.pdf"]):
            main()
        mock_cmd.assert_called_once()

    @patch("pdf2mcp.cli.cmd_stats")
    def test_stats_subcommand_calls_cmd_stats(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "stats"]):
            main()
        mock_cmd.assert_called_once()

    @patch("pdf2mcp.cli.cmd_search")
    def test_search_subcommand_calls_cmd_search(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "search", "test query"]):
            main()
        mock_cmd.assert_called_once()

    @patch("pdf2mcp.cli.cmd_ingest")
    def test_ingest_verbose_flag(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "ingest", "-v"]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.verbose is True

    @patch("pdf2mcp.cli.cmd_ingest")
    def test_ingest_force_flag(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "ingest", "--force"]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.force is True

    @patch("pdf2mcp.cli.cmd_ingest")
    def test_ingest_docs_dir_flag(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "ingest", "--docs-dir", "/some/path"]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.docs_dir == "/some/path"

    @patch("pdf2mcp.cli.cmd_serve")
    def test_serve_transport_defaults_to_none(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "serve"]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.transport is None
        assert args.host is None
        assert args.port is None
        assert args.name is None
        assert args.docs_dir is None

    @patch("pdf2mcp.cli.cmd_serve")
    def test_serve_transport_options_parsed(self, mock_cmd: MagicMock) -> None:
        with patch(
            "sys.argv",
            [
                "pdf2mcp",
                "serve",
                "--transport",
                "streamable-http",
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--name",
                "my-docs",
                "--docs-dir",
                "/my/pdfs",
            ],
        ):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.transport == "streamable-http"
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.name == "my-docs"
        assert args.docs_dir == "/my/pdfs"


# ── cmd_ingest ────────────────────────────────────────────────────


class TestCmdIngest:
    """Test the ingest subcommand."""

    @patch("pdf2mcp.cli.setup_logging")
    def test_config_error_exits_with_code_1(self, mock_logging: MagicMock) -> None:
        args = MagicMock()
        args.verbose = False
        args.force = False
        args.docs_dir = None

        with patch(
            "pdf2mcp.config.get_settings",
            side_effect=ValueError("Missing OPENAI_API_KEY"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_ingest(args)
            assert exc_info.value.code == 1

    @patch("pdf2mcp.ingest.run_ingestion")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_success_calls_run_ingestion(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        settings = MagicMock()
        settings.docs_dir = MagicMock()
        settings.docs_dir.exists.return_value = True
        settings.data_dir = "data"
        mock_settings.return_value = settings

        args = MagicMock()
        args.verbose = False
        args.force = False
        args.docs_dir = None
        cmd_ingest(args)

        mock_run.assert_called_once_with(settings, force=False, show_progress=True)

    @patch("pdf2mcp.ingest.run_ingestion")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_docs_dir_override(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        docs_path = tmp_path / "my-pdfs"
        docs_path.mkdir()

        settings = MagicMock()
        settings.docs_dir = Path("docs")
        settings.data_dir = "data"
        mock_settings.return_value = settings

        args = MagicMock()
        args.verbose = False
        args.force = False
        args.docs_dir = str(docs_path)

        cmd_ingest(args)

        assert settings.docs_dir == docs_path
        mock_run.assert_called_once()

    @patch("pdf2mcp.ingest.run_ingestion", side_effect=RuntimeError("boom"))
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_ingestion_failure_exits_with_code_1(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        settings = MagicMock()
        settings.docs_dir = MagicMock()
        settings.docs_dir.exists.return_value = True
        settings.data_dir = "data"
        mock_settings.return_value = settings

        args = MagicMock()
        args.verbose = False
        args.force = False
        args.docs_dir = None

        with pytest.raises(SystemExit) as exc_info:
            cmd_ingest(args)
        assert exc_info.value.code == 1

    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_missing_docs_dir_exits_with_code_1(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        settings = MagicMock()
        settings.docs_dir = MagicMock()
        settings.docs_dir.exists.return_value = False
        mock_settings.return_value = settings

        args = MagicMock()
        args.verbose = False
        args.force = False
        args.docs_dir = None

        with pytest.raises(SystemExit) as exc_info:
            cmd_ingest(args)
        assert exc_info.value.code == 1


# ── cmd_serve ─────────────────────────────────────────────────────


class TestCmdServe:
    """Test the serve subcommand."""

    @patch("pdf2mcp.config.get_settings", side_effect=ValueError("No key"))
    @patch("pdf2mcp.cli.setup_logging")
    def test_config_error_exits_with_code_1(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        args = MagicMock()
        args.verbose = False

        with pytest.raises(SystemExit) as exc_info:
            cmd_serve(args)
        assert exc_info.value.code == 1

    @patch("pdf2mcp.server.run_server")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_success_calls_run_server(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        settings = MagicMock()
        settings.server_transport = "stdio"
        settings.server_host = "127.0.0.1"
        settings.server_port = 8000
        settings.server_name = "pdf-docs"
        mock_settings.return_value = settings

        args = MagicMock()
        args.verbose = False
        args.transport = None
        args.host = None
        args.port = None
        args.name = None
        args.docs_dir = None

        cmd_serve(args)

        mock_run.assert_called_once_with(
            transport="stdio", host="127.0.0.1", port=8000, name="pdf-docs"
        )

    @patch("pdf2mcp.server.run_server")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_cli_args_override_settings(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        settings = MagicMock()
        settings.server_transport = "stdio"
        settings.server_host = "127.0.0.1"
        settings.server_port = 8000
        settings.server_name = "pdf-docs"
        mock_settings.return_value = settings

        args = MagicMock()
        args.verbose = False
        args.transport = "streamable-http"
        args.host = "0.0.0.0"
        args.port = 9000
        args.name = "my-docs"
        args.docs_dir = None

        cmd_serve(args)

        mock_run.assert_called_once_with(
            transport="streamable-http", host="0.0.0.0", port=9000, name="my-docs"
        )


# ── cmd_config ────────────────────────────────────────────────────


def _config_args(
    *,
    name: str | None = None,
    transport: str | None = None,
    url: str | None = None,
    client: str | None = None,
) -> argparse.Namespace:
    """Build a Namespace for cmd_config with the correct attributes."""
    return argparse.Namespace(name=name, transport=transport, url=url, client=client)


class TestCmdConfig:
    """Test the config subcommand."""

    @pytest.fixture(autouse=True)
    def _set_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure OPENAI_API_KEY is set so get_settings() succeeds."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        from pdf2mcp.config import get_settings

        get_settings.cache_clear()

    def test_prints_all_clients(self, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_config(_config_args())

        output = capsys.readouterr().out
        assert "Claude Code" in output
        assert "Claude Desktop" in output
        assert "Cursor" in output
        assert "VS Code" in output

    def test_single_client(self, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_config(_config_args(client="cursor"))

        output = capsys.readouterr().out
        assert "Cursor" in output
        assert "Claude Code" not in output

    def test_custom_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_config(_config_args(name="my-docs", client="claude-code"))

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        assert "my-docs" in parsed["mcpServers"]

    def test_http_transport(self, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_config(
            _config_args(
                transport="streamable-http",
                url="http://localhost:9000/mcp",
                client="claude-code",
            )
        )

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        server = parsed["mcpServers"]["pdf-docs"]
        assert server["type"] == "http"
        assert "9000" in server["url"]

    def test_stdio_transport(self, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_config(_config_args(transport="stdio", client="claude-code"))

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        server = parsed["mcpServers"]["pdf-docs"]
        assert server["command"] == "uv"
        assert "pdf2mcp" in server["args"]

    def test_vscode_uses_servers_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        cmd_config(_config_args(client="vscode"))

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        assert "servers" in parsed
        assert "mcpServers" not in parsed

    def test_claude_desktop_http_shows_note(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cmd_config(_config_args(transport="streamable-http", client="claude-desktop"))

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        server = parsed["mcpServers"]["pdf-docs"]
        assert server["type"] == "http"
        assert "Claude Desktop requires stdio" in output


# ── cmd_init ──────────────────────────────────────────────────────


class TestCmdInit:
    """Test the init subcommand."""

    def test_creates_docs_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "my-project"
        args = MagicMock()
        args.directory = str(target)
        args.interactive = False

        cmd_init(args)

        assert (target / "docs").is_dir()

    def test_creates_env_file(self, tmp_path: Path) -> None:
        target = tmp_path / "my-project"
        args = MagicMock()
        args.directory = str(target)
        args.interactive = False

        cmd_init(args)

        env_file = target / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "OPENAI_API_KEY" in content
        assert "PDF2MCP_" in content

    def test_skips_existing_env_file(self, tmp_path: Path) -> None:
        target = tmp_path / "my-project"
        target.mkdir(parents=True)
        env_file = target / ".env"
        env_file.write_text("EXISTING=value")

        args = MagicMock()
        args.directory = str(target)
        args.interactive = False

        cmd_init(args)

        assert env_file.read_text() == "EXISTING=value"

    def test_current_dir_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        args.directory = "."
        args.interactive = False

        cmd_init(args)

        assert (tmp_path / "docs").is_dir()
        assert (tmp_path / ".env").exists()


# ── cmd_delete ─────────────────────────────────────────────────────


class TestCmdDelete:
    """Test the delete subcommand."""

    @patch("pdf2mcp.cli.setup_logging")
    def test_config_error_exits_with_code_1(self, mock_logging: MagicMock) -> None:
        args = MagicMock()
        args.verbose = False
        args.filename = "test.pdf"
        args.yes = True

        with patch(
            "pdf2mcp.config.get_settings",
            side_effect=ValueError("Missing OPENAI_API_KEY"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cmd_delete(args)
            assert exc_info.value.code == 1

    @patch("pdf2mcp.store.delete_ingestion_metadata")
    @patch("pdf2mcp.store.delete_by_source")
    @patch("pdf2mcp.store.get_ingested_files")
    @patch("pdf2mcp.store.get_db")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_confirmed_delete(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_get_db: MagicMock,
        mock_ingested: MagicMock,
        mock_delete: MagicMock,
        mock_delete_meta: MagicMock,
    ) -> None:
        mock_settings.return_value = MagicMock()
        mock_ingested.return_value = {"test.pdf": "hash123"}

        args = MagicMock()
        args.verbose = False
        args.filename = "test.pdf"
        args.yes = True

        cmd_delete(args)

        mock_delete.assert_called_once()
        mock_delete_meta.assert_called_once()

    @patch("pdf2mcp.store.get_ingested_files")
    @patch("pdf2mcp.store.get_db")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_not_found_exits_with_code_1(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_get_db: MagicMock,
        mock_ingested: MagicMock,
    ) -> None:
        mock_settings.return_value = MagicMock()
        mock_ingested.return_value = {}

        args = MagicMock()
        args.verbose = False
        args.filename = "nonexistent.pdf"
        args.yes = True

        with pytest.raises(SystemExit) as exc_info:
            cmd_delete(args)
        assert exc_info.value.code == 1

    @patch("builtins.input", return_value="n")
    @patch("pdf2mcp.store.get_ingested_files")
    @patch("pdf2mcp.store.get_db")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_cancellation(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_get_db: MagicMock,
        mock_ingested: MagicMock,
        mock_input: MagicMock,
    ) -> None:
        mock_settings.return_value = MagicMock()
        mock_ingested.return_value = {"test.pdf": "hash123"}

        args = MagicMock()
        args.verbose = False
        args.filename = "test.pdf"
        args.yes = False

        # Should not raise — just cancel
        cmd_delete(args)
        # delete_by_source should NOT have been called
        with patch("pdf2mcp.store.delete_by_source") as mock_del:
            mock_del.assert_not_called()


# ── cmd_stats ──────────────────────────────────────────────────────


class TestCmdStats:
    """Test the stats subcommand."""

    @patch("pdf2mcp.search.list_ingested_documents")
    @patch("pdf2mcp.store.get_db")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_empty_database(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_get_db: MagicMock,
        mock_list: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings = MagicMock()
        settings.data_dir = Path("/tmp/nonexistent")
        settings.docs_dir = Path("docs")
        settings.embedding_model = "text-embedding-3-small"
        settings.search_mode = "semantic"
        mock_settings.return_value = settings
        mock_list.return_value = []

        # Mock table_exists to return False
        with patch("pdf2mcp.store.table_exists", return_value=False):
            args = MagicMock()
            args.verbose = False
            cmd_stats(args)

        output = capsys.readouterr().err
        assert "Documents" in output
        assert "0" in output

    @patch("pdf2mcp.search.list_ingested_documents")
    @patch("pdf2mcp.store.get_db")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_populated_database(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_get_db: MagicMock,
        mock_list: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        settings = MagicMock()
        settings.data_dir = Path("/tmp/nonexistent")
        settings.docs_dir = Path("docs")
        settings.embedding_model = "text-embedding-3-small"
        settings.search_mode = "semantic"
        mock_settings.return_value = settings
        mock_list.return_value = [
            {"filename": "test.pdf", "file_hash": "abc123", "chunk_count": 10},
        ]

        with patch("pdf2mcp.store.table_exists", return_value=False):
            args = MagicMock()
            args.verbose = False
            cmd_stats(args)

        output = capsys.readouterr().err
        assert "test.pdf" in output


# ── cmd_search ─────────────────────────────────────────────────────


class TestCmdSearch:
    """Test the search subcommand."""

    @patch("pdf2mcp.search.search_documents")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_basic_query(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_search: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_settings.return_value = MagicMock()
        mock_search.return_value = []

        args = MagicMock()
        args.verbose = False
        args.query = "test query"
        args.filename = None
        args.num_results = 5

        cmd_search(args)

        mock_search.assert_called_once()
        output = capsys.readouterr().err
        assert "No results found" in output

    @patch("pdf2mcp.search.search_in_document")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_filename_filter(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_search: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_settings.return_value = MagicMock()
        mock_search.return_value = []

        args = MagicMock()
        args.verbose = False
        args.query = "test query"
        args.filename = "manual.pdf"
        args.num_results = 5

        cmd_search(args)

        mock_search.assert_called_once()
        # Verify filename was passed
        call_args = mock_search.call_args
        assert call_args[0][1] == "manual.pdf"

    @patch("pdf2mcp.search.search_documents")
    @patch("pdf2mcp.config.get_settings")
    @patch("pdf2mcp.cli.setup_logging")
    def test_num_results_passed(
        self,
        mock_logging: MagicMock,
        mock_settings: MagicMock,
        mock_search: MagicMock,
    ) -> None:
        mock_settings.return_value = MagicMock()
        mock_search.return_value = []

        args = MagicMock()
        args.verbose = False
        args.query = "test"
        args.filename = None
        args.num_results = 10

        cmd_search(args)

        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs["num_results"] == 10
