from pathlib import Path

from app.rag.ingest import Document

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "docs"


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """The help articles: every .md file in docs_dir, its first heading as
    the title. Used to ingest them, and to serve them at /articles."""
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        title, body = path.stem, text
        first_line, _, rest = text.partition("\n")
        if first_line.startswith("#"):
            # Ingest heads every chunk with the title, so leaving the heading
            # in the body would repeat it in the first chunk.
            title, body = first_line.lstrip("#").strip() or path.stem, rest.strip()
        documents.append(Document(doc_id=path.stem, title=title, text=body))
    return documents
