"""Configuration management using Pydantic Settings.

``ServerSettings`` holds everything the server needs to run: API keys,
storage paths, embedding/chunking parameters, and the network address the
server binds to.

Internal embedding constants are defined at module level and are not
user-configurable:

- ``EMBEDDING_DIMENSIONS`` — vector dimensions produced by the embedding model.
- ``EMBEDDING_BATCH_SIZE`` — maximum number of texts sent per API request.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_DIMENSIONS",
    "ServerSettings",
    "get_settings",
]

# ---------------------------------------------------------------------------
# Internal constants — not exposed as environment-variable settings
# ---------------------------------------------------------------------------

EMBEDDING_DIMENSIONS: int = 1536
EMBEDDING_BATCH_SIZE: int = 2048


# ---------------------------------------------------------------------------
# Server settings — used by the server process only
# ---------------------------------------------------------------------------


class ServerSettings(BaseSettings):
    """Server-side settings loaded from environment variables and .env file.

    These configure the internals of the pdf2mcp server: API keys, storage
    paths, embedding/chunking parameters, and the network address the server
    binds to.  MCP clients never need access to these values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PDF2MCP_",
        extra="ignore",
    )

    # Required — also accepts OPENAI_API_KEY (without prefix)
    openai_api_key: SecretStr = SecretStr("")

    # Paths (relative to project root)
    docs_dir: Path = Path("docs")
    data_dir: Path = Path("data")

    # OpenAI base URL — also accepts OPENAI_BASE_URL (without prefix)
    openai_base_url: str = "https://api.openai.com/v1"

    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # OCR
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_dpi: int = 300

    @field_validator("ocr_dpi")
    @classmethod
    def _validate_ocr_dpi(cls, v: int) -> int:
        if v <= 0:
            msg = f"ocr_dpi must be positive, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("ocr_language")
    @classmethod
    def _validate_ocr_language(cls, v: str) -> str:
        if not v.strip():
            msg = "ocr_language must not be empty"
            raise ValueError(msg)
        return v

    # Search
    default_num_results: int = 5
    search_mode: str = "semantic"

    @field_validator("search_mode")
    @classmethod
    def _validate_search_mode(cls, v: str) -> str:
        allowed = {"semantic", "hybrid", "keyword"}
        if v not in allowed:
            msg = f"search_mode must be one of {allowed}, got '{v}'"
            raise ValueError(msg)
        return v

    # Bind address
    server_name: str = "pdf-docs"
    server_transport: str = "streamable-http"
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    @model_validator(mode="before")
    @classmethod
    def _resolve_openai_settings(cls, values: dict[str, object]) -> dict[str, object]:
        """Resolve OpenAI settings with unprefixed env var fallbacks."""
        # Load .env into os.environ so fallbacks work in mode="before"
        load_dotenv()

        key = values.get("openai_api_key") or os.environ.get(
            "PDF2MCP_OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")
        )
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is required. "
                "Set it via environment variable or in a .env file."
            )
        values["openai_api_key"] = key

        # Resolve base URL with the same fallback pattern
        base_url = values.get("openai_base_url") or os.environ.get(
            "PDF2MCP_OPENAI_BASE_URL", os.environ.get("OPENAI_BASE_URL")
        )
        if base_url:
            values["openai_base_url"] = base_url

        return values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> ServerSettings:
    """Load and return server settings (cached singleton)."""
    return ServerSettings()
