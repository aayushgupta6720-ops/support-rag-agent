import uuid
from dataclasses import dataclass

from qdrant_client.models import PointStruct

from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_documents
from app.rag.qdrant_store import delete_documents_except, delete_stale_chunks, upsert_points

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


def _with_title(title: str, chunk: str) -> str:
    """Head every chunk with its doc's title. Otherwise a chunk from the middle
    of a doc is embedded without knowing what it's about, and the model reading
    it at answer time can't tell which doc it came from."""
    return f"# {title}\n\n{chunk}"


async def ingest_documents(documents: list[Document], *, prune_missing: bool = False) -> int:
    """Chunk, embed, and upsert documents into Qdrant, then delete any chunks
    left over from a longer earlier version of them. With prune_missing,
    `documents` is taken as the whole corpus and every other doc is deleted
    too. Returns chunk count."""
    if not documents:
        return 0

    all_chunks: list[str] = []
    chunk_meta: list[tuple[Document, int]] = []
    chunk_counts: dict[str, int] = {}

    for document in documents:
        chunks = chunk_text(document.text)
        chunk_counts[document.doc_id] = len(chunks)
        for index, chunk in enumerate(chunks):
            all_chunks.append(_with_title(document.title, chunk))
            chunk_meta.append((document, index))

    if all_chunks:
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

    # After the upsert, so a failed embed or upsert leaves the old chunks
    # searchable instead of deleting them first.
    await delete_stale_chunks(chunk_counts)
    if prune_missing:
        await delete_documents_except(list(chunk_counts))

    return len(all_chunks)
