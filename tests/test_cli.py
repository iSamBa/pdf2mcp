"""Tests for pdf2mcp.cli module."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.cli import cmd_config, cmd_init, cmd_ingest, cmd_serve, main, setup_logging


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
        assert "0.1.0" in captured.out

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
                "pdf2mcp", "serve",
                "--transport", "streamable-http",
                "--host", "0.0.0.0",
                "--port", "9000",
                "--name", "my-docs",
                "--docs-dir", "/my/pdfs",
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
        args.progress = False

        cmd_ingest(args)

        mock_run.assert_called_once_with(settings, force=False, show_progress=False)

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


class TestCmdConfig:
    """Test the config subcommand."""

    def test_prints_all_clients(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = MagicMock()
        args.name = None
        args.transport = None
        args.host = None
        args.port = None
        args.client = None

        cmd_config(args)

        output = capsys.readouterr().out
        assert "Claude Code" in output
        assert "Claude Desktop" in output
        assert "Cursor" in output
        assert "VS Code" in output

    def test_single_client(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = MagicMock()
        args.name = None
        args.transport = None
        args.host = None
        args.port = None
        args.client = "cursor"

        cmd_config(args)

        output = capsys.readouterr().out
        assert "Cursor" in output
        assert "Claude Code" not in output

    def test_custom_name(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = MagicMock()
        args.name = "my-docs"
        args.transport = None
        args.host = None
        args.port = None
        args.client = "claude-code"

        cmd_config(args)

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        assert "my-docs" in parsed["mcpServers"]

    def test_http_transport(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = MagicMock()
        args.name = None
        args.transport = "streamable-http"
        args.host = None
        args.port = 9000
        args.client = "claude-code"

        cmd_config(args)

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        server = parsed["mcpServers"]["pdf-docs"]
        assert server["type"] == "http"
        assert "9000" in server["url"]

    def test_stdio_transport(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = MagicMock()
        args.name = None
        args.transport = None
        args.host = None
        args.port = None
        args.client = "claude-code"

        cmd_config(args)

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        server = parsed["mcpServers"]["pdf-docs"]
        assert server["command"] == "uv"
        assert "pdf2mcp" in server["args"]

    def test_vscode_uses_servers_key(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = MagicMock()
        args.name = None
        args.transport = None
        args.host = None
        args.port = None
        args.client = "vscode"

        cmd_config(args)

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        assert "servers" in parsed
        assert "mcpServers" not in parsed

    def test_claude_desktop_always_stdio(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = MagicMock()
        args.name = None
        args.transport = "streamable-http"
        args.host = None
        args.port = None
        args.client = "claude-desktop"

        cmd_config(args)

        output = capsys.readouterr().out
        parsed = _extract_json(output)
        server = parsed["mcpServers"]["pdf-docs"]
        assert server["command"] == "uv"
        assert "stdio" not in output or "Start server manually" in output


# ── cmd_init ──────────────────────────────────────────────────────


class TestCmdInit:
    """Test the init subcommand."""

    def test_creates_docs_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "my-project"
        args = MagicMock()
        args.directory = str(target)

        cmd_init(args)

        assert (target / "docs").is_dir()

    def test_creates_env_file(self, tmp_path: Path) -> None:
        target = tmp_path / "my-project"
        args = MagicMock()
        args.directory = str(target)

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

        cmd_init(args)

        assert env_file.read_text() == "EXISTING=value"

    def test_current_dir_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        args = MagicMock()
        args.directory = "."

        cmd_init(args)

        assert (tmp_path / "docs").is_dir()
        assert (tmp_path / ".env").exists()
