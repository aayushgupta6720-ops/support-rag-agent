"""Ingest markdown docs from data/docs/ into Qdrant.

Usage:
    python -m scripts.ingest
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.ingest import Document, ingest_documents  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def load_documents() -> list[Document]:
    documents = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        first_line = text.splitlines()[0] if text else path.stem
        title = first_line.lstrip("#").strip() or path.stem
        documents.append(Document(doc_id=path.stem, title=title, text=text))
    return documents


async def main() -> None:
    documents = load_documents()
    if not documents:
        print(f"No .md files found in {DOCS_DIR}")
        return

    chunk_count = await ingest_documents(documents)
    print(f"Ingested {len(documents)} documents into {chunk_count} chunks.")


if __name__ == "__main__":
    asyncio.run(main())
