import asyncio

from google.genai import types

from app.core.config import get_settings
from app.core.gemini_client import get_gemini_client


def _embed_sync(texts: list[str], task_type: str) -> types.EmbedContentResponse:
    settings = get_settings()
    return get_gemini_client().models.embed_content(
        model=settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.embedding_dim,
        ),
    )


async def embed_documents(texts: list[str]) -> list[list[float]]:
    response = await asyncio.to_thread(_embed_sync, texts, "RETRIEVAL_DOCUMENT")
    return [embedding.values for embedding in response.embeddings]


async def embed_query_response(text: str) -> types.EmbedContentResponse:
    """Like embed_query, but returns the full response so callers can read
    usage metadata (billable_character_count) for cost tracking."""
    return await asyncio.to_thread(_embed_sync, [text], "RETRIEVAL_QUERY")


async def embed_query(text: str) -> list[float]:
    response = await embed_query_response(text)
    return response.embeddings[0].values
