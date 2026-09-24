from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    ScoredPoint,
    VectorParams,
)

from app.core.config import get_settings

# Payload fields the ingest-time deletes filter on. Qdrant Cloud's strict mode
# can reject filtered updates on unindexed fields, so index them.
_PAYLOAD_INDEXES = {
    "doc_id": PayloadSchemaType.KEYWORD,
    "chunk_index": PayloadSchemaType.INTEGER,
}


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


async def _ensure_payload_indexes() -> None:
    # Checked on every call rather than only at collection creation, so
    # collections created before these indexes existed get them too.
    collection = get_settings().qdrant_collection
    client = get_client()
    existing = (await client.get_collection(collection)).payload_schema
    for field, schema in _PAYLOAD_INDEXES.items():
        if field not in existing:
            await client.create_payload_index(collection, field, schema)


async def upsert_points(points: list[PointStruct]) -> None:
    await ensure_collection()
    await get_client().upsert(collection_name=get_settings().qdrant_collection, points=points)


async def delete_stale_chunks(chunk_counts: dict[str, int]) -> None:
    """For each doc_id, delete its points at chunk_index >= its current chunk
    count: chunks left over from an earlier ingest when the doc was longer."""
    if not chunk_counts:
        return  # an empty `should` matches every point
    await _delete(
        Filter(
            should=[
                Filter(
                    must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                        FieldCondition(key="chunk_index", range=Range(gte=count)),
                    ]
                )
                for doc_id, count in chunk_counts.items()
            ]
        )
    )


async def delete_documents_except(doc_ids: list[str]) -> None:
    """Delete every point whose doc_id isn't in doc_ids."""
    if not doc_ids:
        raise ValueError("refusing to delete every document in the collection")
    await _delete(Filter(must_not=[FieldCondition(key="doc_id", match=MatchAny(any=doc_ids))]))


async def _delete(points_filter: Filter) -> None:
    await ensure_collection()
    await _ensure_payload_indexes()
    await get_client().delete(
        collection_name=get_settings().qdrant_collection, points_selector=points_filter
    )


async def search(vector: list[float], top_k: int) -> list[ScoredPoint]:
    await ensure_collection()
    response = await get_client().query_points(
        collection_name=get_settings().qdrant_collection,
        query=vector,
        limit=top_k,
    )
    return response.points
