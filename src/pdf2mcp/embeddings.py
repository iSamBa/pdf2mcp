"""OpenAI embedding API wrapper with batching and retry."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from pdf2mcp.config import ServerSettings

__all__ = ["embed_texts"]

logger = logging.getLogger(__name__)

# Cache OpenAI client instances keyed on (api_key_hash, base_url)
_client_cache: dict[tuple[str, str], OpenAI] = {}


def _get_client(api_key: str, base_url: str) -> OpenAI:
    """Return a cached OpenAI client for the given credentials."""
    # Hash the API key so we don't store it as a plain dict key
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    cache_key = (key_hash, base_url)

    if cache_key not in _client_cache:
        _client_cache[cache_key] = OpenAI(api_key=api_key, base_url=base_url)

    return _client_cache[cache_key]


@lru_cache(maxsize=128)
def _cached_query_embedding(
    query: str, model: str, api_key_hash: str, base_url: str
) -> tuple[float, ...]:
    """Cache query embeddings to avoid re-embedding identical queries.

    Returns a tuple (hashable) instead of a list.
    """
    # We need the actual client, look it up from the client cache
    cache_key = (api_key_hash, base_url)
    client = _client_cache[cache_key]
    response = client.embeddings.create(input=[query], model=model)
    return tuple(response.data[0].embedding)


def embed_query(query: str, settings: ServerSettings) -> list[float] | None:
    """Embed a single query string with caching.

    Returns None on failure. Uses an LRU cache to avoid re-embedding
    identical queries.
    """
    api_key = settings.openai_api_key.get_secret_value()
    client = _get_client(api_key, settings.openai_base_url)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

    try:
        embedding = _cached_query_embedding(
            query, settings.embedding_model, key_hash, settings.openai_base_url
        )
        return list(embedding)
    except Exception:
        logger.warning("Failed to embed query", exc_info=True)
        return None


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _embed_batch(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Embed a single batch with retry logic."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def embed_texts(
    texts: list[str],
    settings: ServerSettings,
) -> list[list[float]]:
    """Embed a list of texts using OpenAI API with batching.

    Splits texts into batches of ``settings.embedding_batch_size`` and
    calls the OpenAI embeddings endpoint for each batch. Failed calls
    are retried up to 6 times with exponential backoff.
    """
    if not texts:
        return []

    client = _get_client(
        settings.openai_api_key.get_secret_value(),
        settings.openai_base_url,
    )
    all_embeddings: list[list[float]] = []
    batch_size = settings.embedding_batch_size
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_num, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        logger.info(
            "Embedding batch %d/%d (%d texts)",
            batch_num,
            total_batches,
            len(batch),
        )
        embeddings = _embed_batch(client, batch, settings.embedding_model)
        all_embeddings.extend(embeddings)

    logger.info("Embedded %d texts total", len(all_embeddings))
    return all_embeddings
