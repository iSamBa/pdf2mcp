"""Tests for pdf2mcp.embeddings module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pdf2mcp.embeddings import _client_cache, embed_texts


def _make_settings(
    model: str = "text-embedding-3-small",
) -> MagicMock:
    """Create a mock Settings object."""
    settings = MagicMock()
    settings.openai_api_key.get_secret_value.return_value = "sk-test"
    settings.openai_base_url = "https://api.openai.com/v1"
    settings.embedding_model = model
    return settings


def _make_embedding(dim: int = 3) -> list[float]:
    return [0.1] * dim


def _mock_openai_response(embeddings: list[list[float]]) -> MagicMock:
    """Create a mock OpenAI embeddings response."""
    response = MagicMock()
    items = []
    for emb in embeddings:
        item = MagicMock()
        item.embedding = emb
        items.append(item)
    response.data = items
    return response


@pytest.fixture(autouse=True)
def _clear_client_cache() -> None:
    """Clear the OpenAI client cache between tests."""
    _client_cache.clear()


class TestEmbedTexts:
    """Test embedding with batching."""

    @patch("pdf2mcp.embeddings.OpenAI")
    def test_returns_correct_count(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_openai_response(
            [_make_embedding() for _ in range(3)]
        )

        settings = _make_settings()
        result = embed_texts(["a", "b", "c"], settings)

        assert len(result) == 3
        mock_client.embeddings.create.assert_called_once()

    @patch("pdf2mcp.embeddings.EMBEDDING_BATCH_SIZE", 2)
    @patch("pdf2mcp.embeddings.OpenAI")
    def test_batches_correctly(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # 5 texts, batch size 2 → 3 batches (2, 2, 1)
        def side_effect(input: list[str], model: str) -> MagicMock:
            return _mock_openai_response([_make_embedding() for _ in range(len(input))])

        mock_client.embeddings.create.side_effect = side_effect
        settings = _make_settings()

        result = embed_texts(["a", "b", "c", "d", "e"], settings)

        assert len(result) == 5
        assert mock_client.embeddings.create.call_count == 3

    @patch("pdf2mcp.embeddings.OpenAI")
    def test_empty_input(self, mock_openai_cls: MagicMock) -> None:
        settings = _make_settings()
        result = embed_texts([], settings)
        assert result == []
        mock_openai_cls.assert_not_called()

    @patch("pdf2mcp.embeddings.OpenAI")
    def test_uses_correct_api_key(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_openai_response(
            [_make_embedding()]
        )

        settings = _make_settings()
        embed_texts(["hello"], settings)

        mock_openai_cls.assert_called_once_with(api_key="sk-test", base_url="https://api.openai.com/v1")

    @patch("pdf2mcp.embeddings.OpenAI")
    def test_uses_correct_model(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = _mock_openai_response(
            [_make_embedding()]
        )

        settings = _make_settings(model="text-embedding-3-large")
        embed_texts(["hello"], settings)

        mock_client.embeddings.create.assert_called_once_with(
            input=["hello"], model="text-embedding-3-large"
        )

    @patch("tenacity.nap.time.sleep", return_value=None)
    @patch("pdf2mcp.embeddings.OpenAI")
    def test_retries_on_api_error(
        self, mock_openai_cls: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Fail twice, succeed on third try
        mock_client.embeddings.create.side_effect = [
            RuntimeError("API error"),
            RuntimeError("API error"),
            _mock_openai_response([_make_embedding()]),
        ]

        settings = _make_settings()
        result = embed_texts(["hello"], settings)

        assert len(result) == 1
        assert mock_client.embeddings.create.call_count == 3

    @patch("tenacity.nap.time.sleep", return_value=None)
    @patch("pdf2mcp.embeddings.OpenAI")
    def test_raises_after_max_retries(
        self, mock_openai_cls: MagicMock, _mock_sleep: MagicMock
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("API error")

        settings = _make_settings()
        with pytest.raises(RuntimeError, match="API error"):
            embed_texts(["hello"], settings)
