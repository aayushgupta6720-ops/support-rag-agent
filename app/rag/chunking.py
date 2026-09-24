import re

from app.core.config import get_settings

_SEPARATOR = "\n\n"


def chunk_text(text: str) -> list[str]:
    """Pack paragraphs into chunks up to chunk_max_chars, hard-splitting any
    single paragraph that alone exceeds the limit. Each chunk after the first
    opens with up to chunk_overlap_chars from the end of the one before it, so
    text on either side of a boundary keeps some of its context."""
    settings = get_settings()
    max_chars = settings.chunk_max_chars
    overlap = settings.chunk_overlap_chars

    paragraphs = [p.strip() for p in text.split(_SEPARATOR) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if current:
            candidate = f"{current}{_SEPARATOR}{paragraph}"
            if len(candidate) <= max_chars:
                current = candidate
                continue
            chunks.append(current)
            current = ""

        # paragraph opens a new chunk, whether after a full one or a hard split
        if chunks:
            paragraph = _with_overlap(chunks[-1], paragraph, max_chars, overlap)

        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            step = max_chars - overlap
            # Stop before len - overlap: a window starting there would sit
            # entirely inside the previous window's overlap.
            for i in range(0, len(paragraph) - overlap, step):
                chunks.append(paragraph[i : i + max_chars])

    if current:
        chunks.append(current)

    return chunks


def _with_overlap(previous: str, paragraph: str, max_chars: int, overlap: int) -> str:
    """paragraph, opened with the tail of the previous chunk. A paragraph that
    fits in a chunk on its own only gets as much overlap as still fits, rather
    than being pushed into a hard split."""
    limit = overlap
    if len(paragraph) <= max_chars:
        limit = min(overlap, max_chars - len(paragraph) - len(_SEPARATOR))
    tail = _tail_at_word_boundary(previous, limit)
    return f"{tail}{_SEPARATOR}{paragraph}" if tail else paragraph


def _tail_at_word_boundary(text: str, limit: int) -> str:
    """The longest suffix of text, at most limit chars, that starts at a word
    boundary. Empty if there's none, e.g. text with no whitespace near its end."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    start = len(text) - limit
    if not text[start - 1].isspace():
        boundary = re.search(r"\s", text[start:])
        if not boundary:
            return ""
        start += boundary.end()
    return text[start:].strip()
