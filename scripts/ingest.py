"""Ingest markdown docs from data/docs/ into Qdrant.

Usage:
    python -m scripts.ingest
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.documents import DOCS_DIR  # noqa: E402
from app.rag.documents import load_documents as _load_documents  # noqa: E402
from app.rag.ingest import Document, ingest_documents  # noqa: E402


def load_documents() -> list[Document]:
    return _load_documents(DOCS_DIR)


async def main() -> None:
    documents = load_documents()
    if not documents:
        print(f"No .md files found in {DOCS_DIR}")
        return

    # data/docs is the whole corpus: drop anything in Qdrant from docs that
    # have since been deleted here.
    chunk_count = await ingest_documents(documents, prune_missing=True)
    print(f"Ingested {len(documents)} documents into {chunk_count} chunks.")


if __name__ == "__main__":
    asyncio.run(main())
