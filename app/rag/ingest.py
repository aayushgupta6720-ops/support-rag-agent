import uuid
from dataclasses import dataclass

from qdrant_client.models import PointStruct

from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_documents
from app.rag.qdrant_store import upsert_points

# Stable namespace so re-ingesting the same doc/chunk overwrites its old point
# instead of duplicating it.
_POINT_NAMESPACE = uuid.UUID("5c2f9f3e-3b8a-4b6b-9b6b-8f2b1a5b0f1a")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


def _point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{chunk_index}"))


async def ingest_documents(documents: list[Document]) -> int:
    """Chunk, embed, and upsert documents into Qdrant. Returns chunk count."""
    all_chunks: list[str] = []
    chunk_meta: list[tuple[Document, int]] = []

    for document in documents:
        for index, chunk in enumerate(chunk_text(document.text)):
            all_chunks.append(chunk)
            chunk_meta.append((document, index))

    if not all_chunks:
        return 0

    vectors = await embed_documents(all_chunks)

    points = [
        PointStruct(
            id=_point_id(document.doc_id, index),
            vector=vector,
            payload={
                "doc_id": document.doc_id,
                "title": document.title,
                "chunk_index": index,
                "text": chunk,
            },
        )
        for (document, index), vector, chunk in zip(chunk_meta, vectors, all_chunks)
    ]

    await upsert_points(points)
    return len(points)
