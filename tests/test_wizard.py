"""Tests for the interactive wizard flow and .env generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.interactive import (
    WizardCancelledError,
    WizardResult,
    apply_wizard_result,
    generate_env_content,
    run_wizard,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_result(tmp_path: Path) -> WizardResult:
    """Build a WizardResult with all defaults for testing."""
    return WizardResult(
        target_dir=tmp_path / "my-project",
        openai_api_key="sk-test1234567890abcdef",
        openai_base_url="https://api.openai.com/v1",
        docs_dir="docs",
        data_dir="data",
        embedding_model="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=50,
        server_name="pdf-docs",
        server_transport="streamable-http",
        server_host="127.0.0.1",
        server_port=8000,
        ocr_enabled=True,
        ocr_language="eng",
        ocr_dpi=300,
    )


# ---------------------------------------------------------------------------
# WizardResult
# ---------------------------------------------------------------------------


class TestWizardResult:
    """Test the WizardResult dataclass."""

    def test_holds_all_settings(self, tmp_path: Path) -> None:
        result = _default_result(tmp_path)
        assert result.openai_api_key == "sk-test1234567890abcdef"
        assert result.embedding_model == "text-embedding-3-small"
        assert result.ocr_enabled is True
        assert result.server_port == 8000


# ---------------------------------------------------------------------------
# generate_env_content
# ---------------------------------------------------------------------------


class TestGenerateEnvContent:
    """Test .env file generation."""

    def test_includes_api_key_uncommented(
        self, tmp_path: Path
    ) -> None:
        result = _default_result(tmp_path)
        content = generate_env_content(result)
        assert "OPENAI_API_KEY=sk-test1234567890abcdef" in content
        # Should NOT be commented
        for line in content.splitlines():
            if "OPENAI_API_KEY=" in line and line.startswith("#"):
                pytest.fail("API key should not be commented out")

    def test_defaults_are_commented(self, tmp_path: Path) -> None:
        result = _default_result(tmp_path)
        content = generate_env_content(result)
        # All default values should be commented
        assert "# PDF2MCP_DOCS_DIR=docs" in content
        assert "# PDF2MCP_EMBEDDING_MODEL=text-embedding-3-small" in content
        assert "# PDF2MCP_CHUNK_SIZE=500" in content
        assert "# PDF2MCP_SERVER_PORT=8000" in content

    def test_non_defaults_are_uncommented(
        self, tmp_path: Path
    ) -> None:
        result = _default_result(tmp_path)
        result.docs_dir = "my-pdfs"
        result.server_port = 9000
        content = generate_env_content(result)
        assert "PDF2MCP_DOCS_DIR=my-pdfs" in content
        assert "PDF2MCP_SERVER_PORT=9000" in content
        # Verify these are NOT commented
        for line in content.splitlines():
            if "PDF2MCP_DOCS_DIR=my-pdfs" in line:
                assert not line.startswith("#")
            if "PDF2MCP_SERVER_PORT=9000" in line:
                assert not line.startswith("#")

    def test_custom_base_url_uncommented(
        self, tmp_path: Path
    ) -> None:
        result = _default_result(tmp_path)
        result.openai_base_url = "https://custom.api.com/v1"
        content = generate_env_content(result)
        assert (
            "PDF2MCP_OPENAI_BASE_URL=https://custom.api.com/v1"
            in content
        )

    def test_ocr_disabled(self, tmp_path: Path) -> None:
        result = _default_result(tmp_path)
        result.ocr_enabled = False
        content = generate_env_content(result)
        assert "PDF2MCP_OCR_ENABLED=false" in content

    def test_has_section_headers(self, tmp_path: Path) -> None:
        content = generate_env_content(_default_result(tmp_path))
        assert "OpenAI" in content
        assert "Paths" in content
        assert "Embedding" in content
        assert "Server" in content
        assert "OCR" in content


# ---------------------------------------------------------------------------
# run_wizard
# ---------------------------------------------------------------------------


class TestRunWizard:
    """Test the wizard flow with mocked prompts."""

    @patch("pdf2mcp.interactive.print_banner")
    @patch("pdf2mcp.interactive._step_ocr")
    @patch("pdf2mcp.interactive._step_server")
    @patch("pdf2mcp.interactive._step_embedding")
    @patch("pdf2mcp.interactive._step_docs_dir")
    @patch("pdf2mcp.interactive._step_openai")
    @patch("pdf2mcp.interactive._step_project_dir")
    def test_collects_all_defaults(
        self,
        mock_project: MagicMock,
        mock_openai: MagicMock,
        mock_docs: MagicMock,
        mock_embed: MagicMock,
        mock_server: MagicMock,
        mock_ocr: MagicMock,
        mock_banner: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_project.return_value = tmp_path
        mock_openai.return_value = (
            "sk-testkey123456",
            "https://api.openai.com/v1",
        )
        mock_docs.return_value = "docs"
        mock_embed.return_value = (
            "text-embedding-3-small",
            500,
            50,
        )
        mock_server.return_value = (
            "pdf-docs",
            "streamable-http",
            "127.0.0.1",
            8000,
        )
        mock_ocr.return_value = (True, "eng", 300)

        result = run_wizard(tmp_path)

        assert result.target_dir == tmp_path
        assert result.openai_api_key == "sk-testkey123456"
        assert result.docs_dir == "docs"
        assert result.embedding_model == "text-embedding-3-small"
        assert result.server_transport == "streamable-http"
        assert result.ocr_enabled is True

    @patch("pdf2mcp.interactive.print_banner")
    @patch("pdf2mcp.interactive._step_ocr")
    @patch("pdf2mcp.interactive._step_server")
    @patch("pdf2mcp.interactive._step_embedding")
    @patch("pdf2mcp.interactive._step_docs_dir")
    @patch("pdf2mcp.interactive._step_openai")
    @patch("pdf2mcp.interactive._step_project_dir")
    def test_collects_custom_values(
        self,
        mock_project: MagicMock,
        mock_openai: MagicMock,
        mock_docs: MagicMock,
        mock_embed: MagicMock,
        mock_server: MagicMock,
        mock_ocr: MagicMock,
        mock_banner: MagicMock,
        tmp_path: Path,
    ) -> None:
        custom_dir = tmp_path / "custom"
        mock_project.return_value = custom_dir
        mock_openai.return_value = (
            "sk-custom123",
            "https://custom.api/v1",
        )
        mock_docs.return_value = "pdfs"
        mock_embed.return_value = (
            "text-embedding-3-large",
            1000,
            100,
        )
        mock_server.return_value = (
            "my-server",
            "stdio",
            "127.0.0.1",
            8000,
        )
        mock_ocr.return_value = (False, "eng", 300)

        result = run_wizard(tmp_path)

        assert result.target_dir == custom_dir
        assert result.openai_base_url == "https://custom.api/v1"
        assert result.docs_dir == "pdfs"
        assert result.embedding_model == "text-embedding-3-large"
        assert result.chunk_size == 1000
        assert result.server_transport == "stdio"
        assert result.ocr_enabled is False


# ---------------------------------------------------------------------------
# Wizard step functions
# ---------------------------------------------------------------------------


class TestStepOpenai:
    """Test the API key validation in _step_openai."""

    @patch("pdf2mcp.interactive.text_prompt")
    @patch("pdf2mcp.interactive.secret_prompt")
    @patch("pdf2mcp.interactive.print_step")
    def test_rejects_empty_key(
        self,
        mock_step: MagicMock,
        mock_secret: MagicMock,
        mock_text: MagicMock,
    ) -> None:
        from pdf2mcp.interactive import _step_openai

        # First call returns empty, second returns valid
        mock_secret.side_effect = ["", "sk-valid123"]
        mock_text.return_value = "https://api.openai.com/v1"

        key, url = _step_openai()
        assert key == "sk-valid123"
        assert mock_secret.call_count == 2

    @patch("pdf2mcp.interactive.text_prompt")
    @patch("pdf2mcp.interactive.secret_prompt")
    @patch("pdf2mcp.interactive.print_step")
    def test_rejects_invalid_prefix(
        self,
        mock_step: MagicMock,
        mock_secret: MagicMock,
        mock_text: MagicMock,
    ) -> None:
        from pdf2mcp.interactive import _step_openai

        mock_secret.side_effect = ["bad-key", "sk-good123"]
        mock_text.return_value = "https://api.openai.com/v1"

        key, _ = _step_openai()
        assert key == "sk-good123"
        assert mock_secret.call_count == 2


class TestStepServer:
    """Test conditional host/port prompts."""

    @patch("pdf2mcp.interactive.text_prompt")
    @patch("pdf2mcp.interactive.select_prompt")
    @patch("pdf2mcp.interactive.print_step")
    def test_stdio_skips_host_port(
        self,
        mock_step: MagicMock,
        mock_select: MagicMock,
        mock_text: MagicMock,
    ) -> None:
        from pdf2mcp.interactive import _step_server

        # First text_prompt call is for server name
        mock_text.return_value = "pdf-docs"
        mock_select.return_value = "stdio"

        name, transport, host, port = _step_server()
        assert transport == "stdio"
        assert host == "127.0.0.1"
        assert port == 8000
        # text_prompt called once for name, NOT for host/port
        assert mock_text.call_count == 1

    @patch("pdf2mcp.interactive.text_prompt")
    @patch("pdf2mcp.interactive.select_prompt")
    @patch("pdf2mcp.interactive.print_step")
    def test_http_asks_host_port(
        self,
        mock_step: MagicMock,
        mock_select: MagicMock,
        mock_text: MagicMock,
    ) -> None:
        from pdf2mcp.interactive import _step_server

        mock_text.side_effect = ["pdf-docs", "0.0.0.0", "9000"]
        mock_select.return_value = "streamable-http"

        name, transport, host, port = _step_server()
        assert transport == "streamable-http"
        assert host == "0.0.0.0"
        assert port == 9000
        # name + host + port = 3 calls
        assert mock_text.call_count == 3


class TestStepOcr:
    """Test conditional OCR language/DPI prompts."""

    @patch("pdf2mcp.interactive.text_prompt")
    @patch("pdf2mcp.interactive.confirm_prompt")
    @patch("pdf2mcp.interactive.print_step")
    def test_disabled_skips_details(
        self,
        mock_step: MagicMock,
        mock_confirm: MagicMock,
        mock_text: MagicMock,
    ) -> None:
        from pdf2mcp.interactive import _step_ocr

        mock_confirm.return_value = False

        enabled, lang, dpi = _step_ocr()
        assert enabled is False
        assert lang == "eng"  # default
        assert dpi == 300  # default
        mock_text.assert_not_called()

    @patch("pdf2mcp.interactive.text_prompt")
    @patch("pdf2mcp.interactive.confirm_prompt")
    @patch("pdf2mcp.interactive.print_step")
    def test_enabled_asks_details(
        self,
        mock_step: MagicMock,
        mock_confirm: MagicMock,
        mock_text: MagicMock,
    ) -> None:
        from pdf2mcp.interactive import _step_ocr

        mock_confirm.return_value = True
        mock_text.side_effect = ["fra", "600"]

        enabled, lang, dpi = _step_ocr()
        assert enabled is True
        assert lang == "fra"
        assert dpi == 600
        assert mock_text.call_count == 2


# ---------------------------------------------------------------------------
# apply_wizard_result
# ---------------------------------------------------------------------------


class TestApplyWizardResult:
    """Test file/directory creation."""

    @patch("pdf2mcp.interactive.confirm_prompt", return_value=True)
    @patch("pdf2mcp.interactive._print_summary")
    def test_creates_dirs_and_env(
        self,
        mock_summary: MagicMock,
        mock_confirm: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _default_result(tmp_path)
        apply_wizard_result(result)

        target = result.target_dir
        assert target.is_dir()
        assert (target / "docs").is_dir()
        assert (target / ".env").exists()

        content = (target / ".env").read_text()
        assert "OPENAI_API_KEY" in content

    @patch("pdf2mcp.interactive.confirm_prompt", return_value=False)
    @patch("pdf2mcp.interactive._print_summary")
    def test_declined_raises_cancelled(
        self,
        mock_summary: MagicMock,
        mock_confirm: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _default_result(tmp_path)
        with pytest.raises(WizardCancelledError):
            apply_wizard_result(result)

    @patch("pdf2mcp.interactive.confirm_prompt")
    @patch("pdf2mcp.interactive._print_summary")
    def test_existing_env_overwrite_declined(
        self,
        mock_summary: MagicMock,
        mock_confirm: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _default_result(tmp_path)
        target = result.target_dir
        target.mkdir(parents=True)
        env_path = target / ".env"
        env_path.write_text("EXISTING=value")

        # First confirm: "Write these settings?" → True
        # Second confirm: "Overwrite?" → False
        mock_confirm.side_effect = [True, False]

        apply_wizard_result(result)

        # Original .env should be preserved
        assert env_path.read_text() == "EXISTING=value"

    @patch("pdf2mcp.interactive.confirm_prompt")
    @patch("pdf2mcp.interactive._print_summary")
    def test_existing_env_overwrite_accepted(
        self,
        mock_summary: MagicMock,
        mock_confirm: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = _default_result(tmp_path)
        target = result.target_dir
        target.mkdir(parents=True)
        env_path = target / ".env"
        env_path.write_text("EXISTING=value")

        # First confirm: "Write these settings?" → True
        # Second confirm: "Overwrite?" → True
        mock_confirm.side_effect = [True, True]

        apply_wizard_result(result)

        content = env_path.read_text()
        assert "OPENAI_API_KEY" in content
        assert "EXISTING=value" not in content
