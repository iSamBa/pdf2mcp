"""Tests for pdf2mcp.config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from pdf2mcp.config import EMBEDDING_DIMENSIONS, get_settings
from pdf2mcp.config import ServerSettings as Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all PDF2MCP_ and OPENAI env vars and prevent .env loading."""
    for key in list(os.environ):
        if key.startswith("PDF2MCP_") or key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_dotenv():
    """Prevent load_dotenv from loading the real .env file during tests."""
    with patch("pdf2mcp.config.load_dotenv"):
        yield


class TestSettingsDefaults:
    """Test that default values are correct when OPENAI_API_KEY is set."""

    def test_loads_with_openai_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.openai_api_key.get_secret_value() == "sk-test-key"

    def test_default_docs_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.docs_dir == Path("docs")

    def test_default_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.data_dir == Path("data")

    def test_default_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.embedding_model == "text-embedding-3-small"

    def test_default_embedding_dimensions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert EMBEDDING_DIMENSIONS == 1536

    def test_default_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.chunk_size == 500

    def test_default_chunk_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.chunk_overlap == 50

    def test_default_num_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.default_num_results == 5

    def test_default_server_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.server_name == "pdf-docs"


class TestSettingsValidation:
    """Test validation and error handling."""

    def test_raises_when_api_key_missing(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            Settings()

    def test_accepts_prefixed_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PDF2MCP_OPENAI_API_KEY", "sk-prefixed-key")
        settings = Settings()
        assert settings.openai_api_key.get_secret_value() == "sk-prefixed-key"

    def test_prefixed_key_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PDF2MCP_OPENAI_API_KEY", "sk-prefixed")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-unprefixed")
        settings = Settings()
        assert settings.openai_api_key.get_secret_value() == "sk-prefixed"

    def test_api_key_masked_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-key-12345")
        settings = Settings()
        assert "sk-secret-key-12345" not in repr(settings)


class TestOcrSettingsDefaults:
    """Test OCR settings have correct defaults."""

    def test_default_ocr_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.ocr_enabled is True

    def test_default_ocr_language(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.ocr_language == "eng"

    def test_default_ocr_dpi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.ocr_dpi == 300


class TestOcrSettingsOverrides:
    """Test OCR settings can be overridden via env vars."""

    def test_override_ocr_enabled_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OCR_ENABLED", "false")
        settings = Settings()
        assert settings.ocr_enabled is False

    def test_override_ocr_language(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OCR_LANGUAGE", "fra")
        settings = Settings()
        assert settings.ocr_language == "fra"

    def test_override_ocr_dpi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OCR_DPI", "150")
        settings = Settings()
        assert settings.ocr_dpi == 150


class TestOcrSettingsValidation:
    """Test OCR settings validation."""

    def test_ocr_dpi_zero_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OCR_DPI", "0")
        with pytest.raises(ValidationError, match="ocr_dpi must be positive"):
            Settings()

    def test_ocr_dpi_negative_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OCR_DPI", "-1")
        with pytest.raises(ValidationError, match="ocr_dpi must be positive"):
            Settings()

    def test_ocr_language_empty_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OCR_LANGUAGE", "  ")
        with pytest.raises(ValidationError, match="ocr_language must not be empty"):
            Settings()


class TestSettingsOverrides:
    """Test that custom env vars override defaults."""

    def test_override_docs_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_DOCS_DIR", "/custom/docs")
        settings = Settings()
        assert settings.docs_dir == Path("/custom/docs")

    def test_override_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_DATA_DIR", "/custom/data")
        settings = Settings()
        assert settings.data_dir == Path("/custom/data")

    def test_override_chunk_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_CHUNK_SIZE", "1000")
        settings = Settings()
        assert settings.chunk_size == 1000

    def test_override_embedding_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_EMBEDDING_MODEL", "text-embedding-3-large")
        settings = Settings()
        assert settings.embedding_model == "text-embedding-3-large"

    def test_override_num_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_DEFAULT_NUM_RESULTS", "10")
        settings = Settings()
        assert settings.default_num_results == 10

    def test_override_server_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_SERVER_NAME", "my-docs")
        settings = Settings()
        assert settings.server_name == "my-docs"


class TestSettingsPaths:
    """Test that path settings resolve correctly."""

    def test_docs_dir_is_path_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert isinstance(settings.docs_dir, Path)

    def test_data_dir_is_path_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert isinstance(settings.data_dir, Path)


class TestOpenAIBaseURL:
    """Test openai_base_url resolution from env vars."""

    def test_default_is_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = Settings()
        assert settings.openai_base_url == "https://api.openai.com/v1"

    def test_picks_up_prefixed_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OPENAI_BASE_URL", "https://proxy.example.com/v1")
        settings = Settings()
        assert settings.openai_base_url == "https://proxy.example.com/v1"

    def test_picks_up_unprefixed_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://fallback.example.com/v1")
        settings = Settings()
        assert settings.openai_base_url == "https://fallback.example.com/v1"

    def test_prefixed_takes_precedence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("PDF2MCP_OPENAI_BASE_URL", "https://prefixed.example.com/v1")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://unprefixed.example.com/v1")
        settings = Settings()
        assert settings.openai_base_url == "https://prefixed.example.com/v1"


class TestGetSettings:
    """Test the get_settings convenience function."""

    def test_returns_settings_instance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        settings = get_settings()
        assert isinstance(settings, Settings)
        assert settings.openai_api_key.get_secret_value() == "sk-test-key"

    def test_caches_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        first = get_settings()
        second = get_settings()
        assert first is second
