import asyncio

from google.genai import types
from google.genai.errors import ClientError

from app.core.config import get_settings
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    RateLimitedError,
    get_gemini_client,
    is_daily_quota_error,
    retry_overloaded,
)

# Gemini rejects an embed request with more than 100 texts ("at most 100
# requests can be in one batch").
MAX_TEXTS_PER_REQUEST = 100


def _embed_sync(texts: list[str], task_type: str) -> types.EmbedContentResponse:
    settings = get_settings()
    try:
        return retry_overloaded(
            lambda: get_gemini_client().models.embed_content(
                model=settings.embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embedding_dim,
                ),
            )
        )
    except ClientError as exc:
        if is_daily_quota_error(exc):
            raise DailyQuotaExhaustedError(str(exc)) from exc
        if exc.code == 429:
            raise RateLimitedError(str(exc)) from exc
        raise


async def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    # Sequential, not concurrent: parallel batches would just trip the
    # per-minute quota sooner.
    for start in range(0, len(texts), MAX_TEXTS_PER_REQUEST):
        batch = texts[start : start + MAX_TEXTS_PER_REQUEST]
        response = await asyncio.to_thread(_embed_sync, batch, "RETRIEVAL_DOCUMENT")
        vectors.extend(embedding.values for embedding in response.embeddings)
    return vectors


async def embed_query_response(text: str) -> types.EmbedContentResponse:
    """Like embed_query, but returns the full response so callers can read
    usage metadata (billable_character_count) for cost tracking."""
    return await asyncio.to_thread(_embed_sync, [text], "RETRIEVAL_QUERY")


async def embed_query(text: str) -> list[float]:
    response = await embed_query_response(text)
    return response.embeddings[0].values
