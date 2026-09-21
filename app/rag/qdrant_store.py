from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from app.core.config import get_settings


@lru_cache
def get_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


async def ensure_collection() -> None:
    settings = get_settings()
    client = get_client()
    if not await client.collection_exists(settings.qdrant_collection):
        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )


async def upsert_points(points: list[PointStruct]) -> None:
    await ensure_collection()
    await get_client().upsert(collection_name=get_settings().qdrant_collection, points=points)


async def search(vector: list[float], top_k: int) -> list[ScoredPoint]:
    await ensure_collection()
    response = await get_client().query_points(
        collection_name=get_settings().qdrant_collection,
        query=vector,
        limit=top_k,
    )
    return response.points
