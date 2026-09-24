import pytest

import app.rag.chunking as chunking
from app.core.config import Settings

MAX, OVERLAP = 50, 10


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    settings = Settings(_env_file=None, chunk_max_chars=MAX, chunk_overlap_chars=OVERLAP)
    monkeypatch.setattr(chunking, "get_settings", lambda: settings)


def test_empty_or_whitespace_text_gives_no_chunks():
    assert chunking.chunk_text("") == []
    assert chunking.chunk_text("  \n\n \n\n") == []


def test_small_paragraphs_are_packed_together():
    assert chunking.chunk_text("aaa\n\nbbb\n\n  ccc  ") == ["aaa\n\nbbb\n\nccc"]


def test_starts_a_new_chunk_instead_of_exceeding_the_limit():
    first, second = "a" * 30, "b" * 30

    chunks = chunking.chunk_text(f"{first}\n\n{second}")

    assert chunks == [first, second]


def test_oversized_paragraph_is_hard_split_with_overlap():
    paragraph = "".join(chr(ord("a") + i % 26) for i in range(95))

    chunks = chunking.chunk_text(paragraph)

    assert all(len(c) <= MAX for c in chunks)
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-OVERLAP:] == current[:OVERLAP]
    # reassembling without the overlaps gives back the original text
    assert chunks[0] + "".join(c[OVERLAP:] for c in chunks[1:]) == paragraph


def test_hard_split_emits_no_chunk_already_covered_by_the_overlap():
    # 90 chars fit exactly in two windows ([0:50], [40:90]); a third window
    # starting at 80 would only repeat the previous chunk's tail.
    chunks = chunking.chunk_text("x" * 90)

    assert [len(c) for c in chunks] == [50, 50]


TEN_WORDS = "one two three four five six seven eight nine ten"  # 48 chars


def test_new_chunk_opens_with_the_previous_chunks_tail_cut_at_a_word_boundary():
    chunks = chunking.chunk_text(f"{TEN_WORDS}\n\neleven twelve")

    # the last 10 chars are "t nine ten"; the partial "t" is dropped
    assert chunks == [TEN_WORDS, "nine ten\n\neleven twelve"]


def test_overlap_shrinks_rather_than_hard_splitting_a_paragraph_that_fits_alone():
    roomy, tight = "z" * 44, "z" * 47

    # 44 + separator leaves room for 4 chars of overlap ("ten"), 47 for none
    assert chunking.chunk_text(f"{TEN_WORDS}\n\n{roomy}") == [TEN_WORDS, f"ten\n\n{roomy}"]
    assert chunking.chunk_text(f"{TEN_WORDS}\n\n{tight}") == [TEN_WORDS, tight]


def test_paragraph_after_a_hard_split_overlaps_only_the_last_window():
    # the hard split's leftover isn't carried over whole, just the usual tail
    chunks = chunking.chunk_text(f"{'word ' * 14}end\n\ntail")

    assert chunks[-1] == "word end\n\ntail"


def test_no_overlap_when_the_previous_chunk_ends_in_one_long_word():
    chunks = chunking.chunk_text(f"{'x' * 60}\n\ntail")

    assert chunks[-1] == "tail"


def test_every_chunk_respects_the_limit_with_overlap():
    words = " ".join(f"w{i}" for i in range(40))
    text = "\n\n".join([TEN_WORDS, words, "short", TEN_WORDS, "eleven twelve"])

    assert all(len(c) <= MAX for c in chunking.chunk_text(text))
