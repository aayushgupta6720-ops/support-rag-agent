from dataclasses import dataclass

from app.core.config import get_settings
from app.core.observability import time_step
from app.core.pricing import embedding_cost_usd
from app.rag.documents import load_documents
from app.rag.embeddings import embed_query_response
from app.rag.ingest import _with_title
from app.rag.qdrant_store import search


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    text: str
    score: float


async def retrieve(
    query: str,
    top_k: int | None = None,
    min_score: float | None = None,
    max_score_gap: float | None = None,
) -> list[RetrievedChunk]:
    """The top_k chunks most similar to `query`, best first, leaving out any
    below min_score or more than max_score_gap below the best chunk (all
    three default to the settings)."""
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

    min_score = settings.retrieval_min_score if min_score is None else min_score
    max_score_gap = settings.retrieval_max_score_gap if max_score_gap is None else max_score_gap
    if max_score_gap and points:
        min_score = max(min_score, points[0].score - max_score_gap)
    chunks = [
        RetrievedChunk(
            doc_id=point.payload["doc_id"],
            title=point.payload["title"],
            text=point.payload["text"],
            score=point.score,
        )
        for point in points
        if not min_score or point.score >= min_score
    ]
    return whole_articles(chunks) if settings.retrieval_expand_docs else chunks


def whole_articles(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """One entry per article the chunks came from, best first, holding the
    whole article under its title, as each chunk is. An article whose text
    isn't on disk keeps its chunks."""
    articles = {doc.doc_id: doc for doc in load_documents()}
    out: list[RetrievedChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        doc = articles.get(chunk.doc_id)
        if doc is None:
            out.append(chunk)
        elif chunk.doc_id not in seen:
            seen.add(chunk.doc_id)
            out.append(RetrievedChunk(doc.doc_id, doc.title, _with_title(doc.title, doc.text), chunk.score))
    return out
