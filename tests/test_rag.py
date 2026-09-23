from types import SimpleNamespace

import app.rag.ingest as ingest
import app.rag.retrieval as retrieval
from app.core.config import get_settings
from app.core.observability import start_trace
from app.core.pricing import embedding_cost_usd


def _embed_response(values, billable_chars):
    metadata = SimpleNamespace(billable_character_count=billable_chars) if billable_chars else None
    return SimpleNamespace(embeddings=[SimpleNamespace(values=values)], metadata=metadata)


def _point(doc_id, score):
    payload = {"doc_id": doc_id, "title": doc_id.title(), "text": f"{doc_id} text"}
    return SimpleNamespace(payload=payload, score=score)


async def test_retrieve_embeds_query_and_maps_points(monkeypatch):
    searched = {}

    async def fake_embed(text):
        return _embed_response([0.1, 0.2], billable_chars=42)

    async def fake_search(vector, top_k):
        searched.update(vector=vector, top_k=top_k)
        return [_point("billing-refunds", 0.9), _point("account-deletion", 0.5)]

    monkeypatch.setattr(retrieval, "embed_query_response", fake_embed)
    monkeypatch.setattr(retrieval, "search", fake_search)
    trace = start_trace()

    chunks = await retrieval.retrieve("refund?")

    assert searched == {"vector": [0.1, 0.2], "top_k": get_settings().retrieval_top_k}
    assert [(c.doc_id, c.score) for c in chunks] == [("billing-refunds", 0.9), ("account-deletion", 0.5)]
    assert chunks[0].text == "billing-refunds text"
    assert [s.name for s in trace.steps] == ["embed_query", "qdrant_search"]
    assert trace.steps[0].cost_usd == embedding_cost_usd(get_settings().embedding_model, 42)


async def test_retrieve_cost_falls_back_to_query_length_without_metadata(monkeypatch):
    async def fake_embed(text):
        return _embed_response([0.0], billable_chars=None)

    async def fake_search(vector, top_k):
        return []

    monkeypatch.setattr(retrieval, "embed_query_response", fake_embed)
    monkeypatch.setattr(retrieval, "search", fake_search)
    trace = start_trace()

    assert await retrieval.retrieve("twelve chars", top_k=2) == []
    assert trace.steps[0].cost_usd == embedding_cost_usd(get_settings().embedding_model, 12)


def test_point_ids_are_stable_across_runs_and_unique_per_chunk():
    # stable ids are what make re-ingesting overwrite instead of duplicate
    assert ingest._point_id("password-reset", 0) == ingest._point_id("password-reset", 0)
    assert ingest._point_id("password-reset", 0) != ingest._point_id("password-reset", 1)
    assert ingest._point_id("password-reset", 0) != ingest._point_id("billing-refunds", 0)


async def test_ingest_embeds_every_chunk_and_upserts_with_payload(monkeypatch):
    embedded, upserted = [], []

    async def fake_embed(texts):
        embedded.extend(texts)
        return [[float(i)] for i in range(len(texts))]

    async def fake_upsert(points):
        upserted.extend(points)

    monkeypatch.setattr(ingest, "chunk_text", lambda text: text.split("|"))
    monkeypatch.setattr(ingest, "embed_documents", fake_embed)
    monkeypatch.setattr(ingest, "upsert_points", fake_upsert)

    count = await ingest.ingest_documents([
        ingest.Document(doc_id="a", title="A", text="a0|a1"),
        ingest.Document(doc_id="b", title="B", text="b0"),
    ])

    assert count == 3
    assert embedded == ["a0", "a1", "b0"]
    assert [(p.payload["doc_id"], p.payload["chunk_index"], p.payload["text"]) for p in upserted] == [
        ("a", 0, "a0"), ("a", 1, "a1"), ("b", 0, "b0"),
    ]
    assert [p.vector for p in upserted] == [[0.0], [1.0], [2.0]]
    assert upserted[1].id == ingest._point_id("a", 1)


async def test_ingest_with_no_chunks_skips_embedding(monkeypatch):
    async def fail(*args):
        raise AssertionError("should not be called")

    monkeypatch.setattr(ingest, "embed_documents", fail)
    monkeypatch.setattr(ingest, "upsert_points", fail)

    assert await ingest.ingest_documents([]) == 0
