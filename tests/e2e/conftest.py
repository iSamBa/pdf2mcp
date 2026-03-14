"""Shared fixtures for E2E tests.

Provides mock embeddings, synthetic PDFs, project scaffolding,
and cache cleanup so E2E tests exercise the full pipeline with
real LanceDB and real PDF parsing, only mocking OpenAI embeddings.
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pymupdf
import pytest

from pdf2mcp.config import EMBEDDING_DIMENSIONS, ServerSettings

# ---------------------------------------------------------------------------
# a) Deterministic fake embeddings (session-scoped, autouse)
# ---------------------------------------------------------------------------

_RNG = random.Random(42)  # noqa: S311


def _fake_embedding() -> list[float]:
    """Return a deterministic 1536-dim vector."""
    return [_RNG.gauss(0, 1) for _ in range(EMBEDDING_DIMENSIONS)]


def _fake_embed_texts(
    texts: list[str],
    settings: ServerSettings,
    on_batch_complete: Any = None,
) -> list[list[float]]:
    return [_fake_embedding() for _ in texts]


def _fake_embed_query(query: str, settings: ServerSettings) -> list[float] | None:
    return _fake_embedding()


@pytest.fixture(autouse=True, scope="session")
def _mock_embeddings():
    """Patch embedding functions for the entire E2E test session."""
    with (
        patch("pdf2mcp.embeddings.embed_texts", side_effect=_fake_embed_texts),
        patch("pdf2mcp.embeddings.embed_query", side_effect=_fake_embed_query),
        patch("pdf2mcp.ingest.embed_texts", side_effect=_fake_embed_texts),
        patch("pdf2mcp.search.embed_query", side_effect=_fake_embed_query),
    ):
        yield


# ---------------------------------------------------------------------------
# b) Synthetic PDF fixture (session-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a 3-page synthetic PDF with recognisable content."""
    pdf_path = tmp_path_factory.mktemp("pdfs") / "test.pdf"

    doc = pymupdf.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Test Document\n\n# Introduction\n\n"
        "This document describes the pdf2mcp project. "
        "It converts PDF folders into searchable MCP servers. "
        "The system uses OpenAI embeddings for semantic search.",
        fontsize=11,
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "# Safety Guidelines\n\n"
        "Always follow safety procedures when handling equipment. "
        "Wear protective gear including gloves and goggles. "
        "Report any incidents to the safety officer immediately. "
        "Emergency exits are marked with green signs.",
        fontsize=11,
    )

    # Page 3
    page3 = doc.new_page()
    page3.insert_text(
        (72, 72),
        "# Technical Specifications\n\n"
        "Maximum operating temperature: 85 degrees Celsius. "
        "Power consumption: 500 watts nominal. "
        "Weight: 12.5 kilograms. "
        "Dimensions: 300mm x 200mm x 150mm.",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ---------------------------------------------------------------------------
# c) Cache cleanup helpers
# ---------------------------------------------------------------------------


def _clear_all_caches() -> None:
    """Clear all module-level caches."""
    from pdf2mcp.config import get_settings
    from pdf2mcp.embeddings import _cached_query_embedding, _client_cache
    from pdf2mcp.store import _cached_connect, invalidate_table_cache

    get_settings.cache_clear()
    _cached_connect.cache_clear()
    invalidate_table_cache()
    _cached_query_embedding.cache_clear()
    _client_cache.clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear all caches before and after each test."""
    _clear_all_caches()
    yield
    _clear_all_caches()


# ---------------------------------------------------------------------------
# d) E2E project fixture (function-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture()
def e2e_project(
    tmp_path: Path, synthetic_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Set up a complete project directory with PDF and env vars."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    shutil.copy(synthetic_pdf, docs_dir / "test.pdf")

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-fake-e2e-key\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PDF2MCP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PDF2MCP_DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-e2e-key")

    return tmp_path


# ---------------------------------------------------------------------------
# e) Ingested project fixture (function-scoped)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ingested_project(e2e_project: Path) -> Path:
    """E2E project with data already ingested."""
    from pdf2mcp.config import get_settings
    from pdf2mcp.ingest import run_ingestion

    settings = get_settings()
    run_ingestion(settings)
    return e2e_project


# ---------------------------------------------------------------------------
# f) Session-scoped ingested project (for MCP server tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ingested_project_session(
    synthetic_pdf: Path,
    tmp_path_factory: pytest.TempPathFactory,
    _mock_embeddings: None,
) -> Path:
    """Session-scoped project with ingested data.

    Used by MCP server tests that need a long-lived data directory.
    Environment variables are set via os.environ (not monkeypatch)
    because monkeypatch is function-scoped.
    """
    project_dir = tmp_path_factory.mktemp("e2e_server")
    docs_dir = project_dir / "docs"
    docs_dir.mkdir()
    shutil.copy(synthetic_pdf, docs_dir / "test.pdf")

    env_file = project_dir / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-fake-e2e-key\n")

    # Temporarily set env vars for ingestion
    old_env = {}
    env_vars = {
        "PDF2MCP_DATA_DIR": str(project_dir / "data"),
        "PDF2MCP_DOCS_DIR": str(docs_dir),
        "OPENAI_API_KEY": "sk-fake-e2e-key",
    }
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    _clear_all_caches()

    from pdf2mcp.config import get_settings
    from pdf2mcp.ingest import run_ingestion

    settings = get_settings()
    run_ingestion(settings)

    _clear_all_caches()

    yield project_dir

    # Restore env
    for key, old_value in old_env.items():
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value
    _clear_all_caches()
