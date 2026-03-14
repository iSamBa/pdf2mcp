"""Tests for post-setup actions (ingestion offer & config generation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.interactive import (
    CLIENT_CHOICES,
    CLIENT_FILES,
    WizardCancelledError,
    WizardResult,
    build_config_snippet,
    run_post_setup,
    wizard_result_to_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(target_dir: Path | None = None) -> WizardResult:
    """Create a WizardResult with sensible defaults for testing."""
    return WizardResult(
        target_dir=target_dir or Path("/tmp/test-project"),
        openai_api_key="sk-test-key-1234567890",
        openai_base_url="https://api.openai.com/v1",
        docs_dir="docs",
        data_dir="data",
        embedding_model="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=50,
        search_mode="semantic",
        server_name="pdf-docs",
        server_transport="streamable-http",
        server_host="127.0.0.1",
        server_port=8000,
        ocr_enabled=True,
        ocr_language="eng",
        ocr_dpi=300,
    )


# ---------------------------------------------------------------------------
# build_config_snippet
# ---------------------------------------------------------------------------


class TestBuildConfigSnippet:
    """Test the build_config_snippet function."""

    def test_http_transport(self) -> None:
        snippet = build_config_snippet(
            "claude-code",
            "my-server",
            "streamable-http",
            "http://127.0.0.1:8000/mcp",
        )
        assert snippet == {
            "mcpServers": {
                "my-server": {
                    "type": "http",
                    "url": "http://127.0.0.1:8000/mcp",
                },
            },
        }

    def test_stdio_transport(self) -> None:
        snippet = build_config_snippet(
            "claude-desktop",
            "my-server",
            "stdio",
            "",
        )
        assert snippet == {
            "mcpServers": {
                "my-server": {
                    "command": "uv",
                    "args": ["run", "pdf2mcp", "serve"],
                },
            },
        }

    def test_vscode_uses_servers_key(self) -> None:
        snippet = build_config_snippet(
            "vscode",
            "my-server",
            "streamable-http",
            "http://127.0.0.1:8000/mcp",
        )
        assert "servers" in snippet
        assert "mcpServers" not in snippet

    def test_all_clients_have_labels_and_files(self) -> None:
        labels = dict(CLIENT_CHOICES)
        for value, _ in CLIENT_CHOICES:
            assert value in labels
            assert value in CLIENT_FILES


# ---------------------------------------------------------------------------
# wizard_result_to_settings
# ---------------------------------------------------------------------------


class TestWizardResultToSettings:
    """Test wizard_result_to_settings conversion."""

    def test_produces_valid_settings(self) -> None:
        result = _make_result(Path("/tmp/proj"))
        settings = wizard_result_to_settings(result)

        assert settings.docs_dir == Path("/tmp/proj/docs")
        assert settings.data_dir == Path("/tmp/proj/data")
        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.server_name == "pdf-docs"
        assert settings.openai_api_key.get_secret_value() == "sk-test-key-1234567890"

    def test_paths_are_absolute(self) -> None:
        result = _make_result(Path("/my/project"))
        settings = wizard_result_to_settings(result)
        assert settings.docs_dir.is_absolute()
        assert settings.data_dir.is_absolute()


# ---------------------------------------------------------------------------
# _post_setup_ingest (via run_post_setup)
# ---------------------------------------------------------------------------


class TestPostSetupIngest:
    """Test the ingestion offer in post-setup."""

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_no_pdfs_skips_ingestion_offer(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        (tmp_path / "docs").mkdir()

        # confirm_prompt will be called for config step — decline it
        mock_confirm.return_value = False

        run_post_setup(result)

        # Should print "No PDFs found" message
        output = " ".join(str(call) for call in mock_console.print.call_args_list)
        assert "No PDFs" in output or "no pdfs" in output.lower()

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.ingest.run_ingestion")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_pdfs_found_user_accepts_runs_ingestion(
        self,
        mock_confirm: MagicMock,
        mock_run_ingestion: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "test.pdf").write_bytes(b"%PDF-1.4 test")

        # First confirm: ingest? Yes. Then config? No.
        mock_confirm.side_effect = [True, False]

        run_post_setup(result)

        mock_run_ingestion.assert_called_once()
        call_kwargs = mock_run_ingestion.call_args
        assert call_kwargs[1]["show_progress"] is True

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_pdfs_found_user_declines_ingestion(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "test.pdf").write_bytes(b"%PDF-1.4 test")

        # First confirm: ingest? No. Then config? No.
        mock_confirm.side_effect = [False, False]

        run_post_setup(result)

        output = " ".join(str(call) for call in mock_console.print.call_args_list)
        assert "pdf2mcp ingest" in output

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_ingestion_error_caught_gracefully(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "test.pdf").write_bytes(b"%PDF-1.4 test")

        # Ingest? Yes. Config? No.
        mock_confirm.side_effect = [True, False]

        with patch(
            "pdf2mcp.ingest.run_ingestion",
            side_effect=RuntimeError("embedding API down"),
        ):
            # Should not raise
            run_post_setup(result)

        output = " ".join(str(call) for call in mock_console.print.call_args_list)
        assert "failed" in output.lower()


# ---------------------------------------------------------------------------
# _post_setup_config (via run_post_setup)
# ---------------------------------------------------------------------------


class TestPostSetupConfig:
    """Test the config generation in post-setup."""

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_user_declines_config(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        (tmp_path / "docs").mkdir()

        # No PDFs (skip ingest offer), decline config
        mock_confirm.return_value = False

        run_post_setup(result)

        output = " ".join(str(call) for call in mock_console.print.call_args_list)
        assert "pdf2mcp config" in output

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_config_snippets_generated_for_selected_clients(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        from rich.panel import Panel

        result = _make_result(tmp_path)
        (tmp_path / "docs").mkdir()

        # No PDFs → skip ingest. Config? Yes. Claude Code? Yes. Others? No.
        mock_confirm.side_effect = [True, True, False, False, False]

        run_post_setup(result)

        # Find a Panel argument in the print calls
        panels = [
            call.args[0]
            for call in mock_console.print.call_args_list
            if call.args and isinstance(call.args[0], Panel)
        ]
        assert len(panels) == 1
        assert "Claude Code" in panels[0].title

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_no_clients_selected(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        (tmp_path / "docs").mkdir()

        # No PDFs. Config? Yes. All clients? No.
        mock_confirm.side_effect = [True, False, False, False, False]

        run_post_setup(result)

        output = " ".join(str(call) for call in mock_console.print.call_args_list)
        assert "No clients selected" in output


# ---------------------------------------------------------------------------
# Ctrl+C handling
# ---------------------------------------------------------------------------


class TestPostSetupCtrlC:
    """Test that Ctrl+C during post-setup raises WizardCancelledError."""

    @patch("pdf2mcp.interactive._console")
    @patch(
        "pdf2mcp.interactive.confirm_prompt",
        side_effect=WizardCancelledError,
    )
    def test_ctrl_c_during_ingest_prompt(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "test.pdf").write_bytes(b"%PDF-1.4 test")

        with pytest.raises(WizardCancelledError):
            run_post_setup(result)

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt")
    def test_ctrl_c_during_config_prompt(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        (tmp_path / "docs").mkdir()

        # No PDFs → skip ingest. Config prompt → Ctrl+C
        mock_confirm.side_effect = [WizardCancelledError]

        with pytest.raises(WizardCancelledError):
            run_post_setup(result)


# ---------------------------------------------------------------------------
# User declines both actions
# ---------------------------------------------------------------------------


class TestPostSetupDeclineBoth:
    """Test user declining both post-setup actions."""

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt", return_value=False)
    def test_decline_both_no_pdfs(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        (tmp_path / "docs").mkdir()

        # Should complete without error
        run_post_setup(result)

        # confirm only called once (for config, since no PDFs means no ingest prompt)
        assert mock_confirm.call_count == 1

    @patch("pdf2mcp.interactive._console")
    @patch("pdf2mcp.interactive.confirm_prompt", return_value=False)
    def test_decline_both_with_pdfs(
        self,
        mock_confirm: MagicMock,
        mock_console: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _make_result(tmp_path)
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "test.pdf").write_bytes(b"%PDF-1.4 test")

        run_post_setup(result)

        # confirm called twice: ingest + config
        assert mock_confirm.call_count == 2
