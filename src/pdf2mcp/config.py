"""Configuration management using Pydantic Settings."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PDF2MCP_",
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
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 2048

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Search
    default_num_results: int = 5

    # Server
    server_name: str = "pdf-docs"
    server_transport: str = "stdio"
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and return application settings (cached singleton)."""
    return Settings()
