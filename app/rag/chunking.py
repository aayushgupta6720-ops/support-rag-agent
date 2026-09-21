from app.core.config import get_settings


def chunk_text(text: str) -> list[str]:
    """Pack paragraphs into chunks up to chunk_max_chars, hard-splitting any
    single paragraph that alone exceeds the limit."""
    settings = get_settings()
    max_chars = settings.chunk_max_chars
    overlap = settings.chunk_overlap_chars

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            step = max_chars - overlap
            for i in range(0, len(paragraph), step):
                chunks.append(paragraph[i : i + max_chars])

    if current:
        chunks.append(current)

    return chunks
