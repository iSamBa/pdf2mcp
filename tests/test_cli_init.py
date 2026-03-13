"""Tests for the --interactive flag on the init subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.cli import cmd_init, main

# ── Flag parsing ─────────────────────────────────────────────────


class TestInitFlagParsing:
    """Verify --interactive / -i are recognized by argparse."""

    @patch("pdf2mcp.cli.cmd_init")
    def test_interactive_long_flag(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "init", "--interactive"]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.interactive is True

    @patch("pdf2mcp.cli.cmd_init")
    def test_interactive_short_flag(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "init", "-i"]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.interactive is True

    @patch("pdf2mcp.cli.cmd_init")
    def test_no_flag_defaults_false(self, mock_cmd: MagicMock) -> None:
        with patch("sys.argv", ["pdf2mcp", "init"]):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.interactive is False

    @patch("pdf2mcp.cli.cmd_init")
    def test_directory_with_interactive(self, mock_cmd: MagicMock) -> None:
        with patch(
            "sys.argv",
            ["pdf2mcp", "init", "mydir", "--interactive"],
        ):
            main()
        args = mock_cmd.call_args[0][0]
        assert args.directory == "mydir"
        assert args.interactive is True


# ── Command dispatch ─────────────────────────────────────────────


class TestCmdInitDispatch:
    """Verify cmd_init dispatches to the correct path."""

    @patch("pdf2mcp.cli._cmd_init_scaffold")
    def test_no_flag_calls_scaffold(self, mock_scaffold: MagicMock) -> None:
        args = MagicMock()
        args.interactive = False
        args.directory = "."

        cmd_init(args)
        mock_scaffold.assert_called_once_with(args)

    @patch("pdf2mcp.cli._cmd_init_interactive")
    def test_flag_calls_interactive(self, mock_interactive: MagicMock) -> None:
        args = MagicMock()
        args.interactive = True
        args.directory = "."

        cmd_init(args)
        mock_interactive.assert_called_once_with(args)


# ── Scaffold path unchanged ──────────────────────────────────────


class TestScaffoldUnchanged:
    """Verify the non-interactive path works as before."""

    def test_creates_docs_dir(self, tmp_path: Path) -> None:
        args = MagicMock()
        args.interactive = False
        args.directory = str(tmp_path / "project")

        cmd_init(args)

        assert (tmp_path / "project" / "docs").is_dir()

    def test_creates_env_file(self, tmp_path: Path) -> None:
        args = MagicMock()
        args.interactive = False
        args.directory = str(tmp_path / "project")

        cmd_init(args)

        env_file = tmp_path / "project" / ".env"
        assert env_file.exists()
        assert "OPENAI_API_KEY" in env_file.read_text()


# ── Interactive path ─────────────────────────────────────────────


class TestInteractivePath:
    """Verify the wizard is invoked and errors are handled."""

    @patch("pdf2mcp.interactive.run_post_setup")
    @patch("pdf2mcp.interactive.apply_wizard_result")
    @patch("pdf2mcp.interactive.run_wizard")
    def test_calls_wizard(
        self,
        mock_wizard: MagicMock,
        mock_apply: MagicMock,
        mock_post_setup: MagicMock,
    ) -> None:
        mock_wizard.return_value = MagicMock()
        args = MagicMock()
        args.interactive = True
        args.directory = "/tmp/test"

        cmd_init(args)

        mock_wizard.assert_called_once_with(Path("/tmp/test"))
        mock_apply.assert_called_once_with(mock_wizard.return_value)
        mock_post_setup.assert_called_once_with(mock_wizard.return_value)

    @patch("pdf2mcp.interactive.run_wizard")
    def test_cancelled_exits_130(self, mock_wizard: MagicMock) -> None:
        from pdf2mcp.interactive import WizardCancelledError

        mock_wizard.side_effect = WizardCancelledError

        args = MagicMock()
        args.interactive = True
        args.directory = "."

        with pytest.raises(SystemExit) as exc_info:
            cmd_init(args)
        assert exc_info.value.code == 130

    @patch("pdf2mcp.interactive.apply_wizard_result")
    @patch("pdf2mcp.interactive.run_wizard")
    def test_apply_cancelled_exits_130(
        self,
        mock_wizard: MagicMock,
        mock_apply: MagicMock,
    ) -> None:
        from pdf2mcp.interactive import WizardCancelledError

        mock_wizard.return_value = MagicMock()
        mock_apply.side_effect = WizardCancelledError

        args = MagicMock()
        args.interactive = True
        args.directory = "."

        with pytest.raises(SystemExit) as exc_info:
            cmd_init(args)
        assert exc_info.value.code == 130
