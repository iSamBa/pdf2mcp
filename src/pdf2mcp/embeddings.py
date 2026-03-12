"""OpenAI embedding API wrapper with batching and retry."""

from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from pdf2mcp.config import Settings

__all__ = ["embed_texts"]

logger = logging.getLogger(__name__)


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
    settings: Settings,
) -> list[list[float]]:
    """Embed a list of texts using OpenAI API with batching.

    Splits texts into batches of ``settings.embedding_batch_size`` and
    calls the OpenAI embeddings endpoint for each batch. Failed calls
    are retried up to 6 times with exponential backoff.
    """
    if not texts:
        return []

    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
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
