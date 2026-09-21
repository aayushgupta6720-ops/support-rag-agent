from dataclasses import dataclass

from app.core.config import get_settings
from app.core.observability import time_step
from app.core.pricing import embedding_cost_usd
from app.rag.embeddings import embed_query_response
from app.rag.qdrant_store import search


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    text: str
    score: float


async def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    settings = get_settings()

    with time_step("embed_query") as usage:
        embed_response = await embed_query_response(query)
        metadata = embed_response.metadata
        # The API doesn't always populate metadata; fall back to the raw
        # character count so cost tracking still gets a reasonable estimate.
        billable_chars = (metadata and metadata.billable_character_count) or len(query)
        usage["cost_usd"] = embedding_cost_usd(settings.embedding_model, billable_chars)
    vector = embed_response.embeddings[0].values

    with time_step("qdrant_search"):
        points = await search(vector, top_k or settings.retrieval_top_k)

    return [
        RetrievedChunk(
            doc_id=point.payload["doc_id"],
            title=point.payload["title"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in points
    ]
