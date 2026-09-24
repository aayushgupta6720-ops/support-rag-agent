from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError, ServerError
from qdrant_client import AsyncQdrantClient

import app.rag.embeddings as embeddings
import app.rag.ingest as ingest
import app.rag.qdrant_store as qdrant_store
import app.rag.retrieval as retrieval
import scripts.ingest as ingest_script
from app.core.config import get_settings
from app.core.gemini_client import DailyQuotaExhaustedError, ModelOverloadedError, RateLimitedError
from app.core.observability import start_trace
from app.core.pricing import embedding_cost_usd
from tests.fakes import DAILY, PER_MINUTE, rate_limited_error


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
    embedded, upserted, stale_checked = [], [], []

    async def fake_embed(texts):
        embedded.extend(texts)
        return [[float(i)] for i in range(len(texts))]

    async def fake_upsert(points):
        upserted.extend(points)

    async def fake_delete_stale(chunk_counts):
        stale_checked.append(chunk_counts)

    monkeypatch.setattr(ingest, "chunk_text", lambda text: text.split("|"))
    monkeypatch.setattr(ingest, "embed_documents", fake_embed)
    monkeypatch.setattr(ingest, "upsert_points", fake_upsert)
    monkeypatch.setattr(ingest, "delete_stale_chunks", fake_delete_stale)

    count = await ingest.ingest_documents([
        ingest.Document(doc_id="a", title="A", text="a0|a1"),
        ingest.Document(doc_id="b", title="B", text="b0"),
    ])

    assert count == 3
    # every chunk, not just the first, carries its doc's title
    assert embedded == ["# A\n\na0", "# A\n\na1", "# B\n\nb0"]
    assert [(p.payload["doc_id"], p.payload["chunk_index"], p.payload["text"]) for p in upserted] == [
        ("a", 0, "# A\n\na0"), ("a", 1, "# A\n\na1"), ("b", 0, "# B\n\nb0"),
    ]
    assert [p.vector for p in upserted] == [[0.0], [1.0], [2.0]]
    assert upserted[1].id == ingest._point_id("a", 1)
    assert stale_checked == [{"a": 2, "b": 1}]


async def test_ingest_with_no_documents_touches_nothing(monkeypatch):
    async def fail(*args):
        raise AssertionError("should not be called")

    monkeypatch.setattr(ingest, "embed_documents", fail)
    monkeypatch.setattr(ingest, "upsert_points", fail)

    assert await ingest.ingest_documents([], prune_missing=True) == 0


async def test_doc_that_is_now_empty_still_has_its_old_chunks_deleted(monkeypatch):
    stale_checked = []

    async def fail(*args):
        raise AssertionError("should not be called")

    async def fake_delete_stale(chunk_counts):
        stale_checked.append(chunk_counts)

    monkeypatch.setattr(ingest, "embed_documents", fail)
    monkeypatch.setattr(ingest, "upsert_points", fail)
    monkeypatch.setattr(ingest, "delete_stale_chunks", fake_delete_stale)

    assert await ingest.ingest_documents([ingest.Document(doc_id="a", title="A", text="  ")]) == 0
    assert stale_checked == [{"a": 0}]


@pytest.fixture
def qdrant(monkeypatch):
    """An in-process Qdrant, so ingest's filtered deletes run against real
    filter semantics without a server. Embeddings are faked."""
    client = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(qdrant_store, "get_client", lambda: client)

    async def fake_embed(texts):
        return [[1.0] * get_settings().embedding_dim for _ in texts]

    monkeypatch.setattr(ingest, "embed_documents", fake_embed)
    monkeypatch.setattr(ingest, "chunk_text", lambda text: text.split("|"))
    return client


async def _stored_chunks(client):
    points, _ = await client.scroll(get_settings().qdrant_collection, limit=100)
    return sorted((p.payload["doc_id"], p.payload["chunk_index"]) for p in points)


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_reingesting_a_shorter_doc_deletes_its_leftover_chunks(qdrant):
    await ingest.ingest_documents([
        ingest.Document(doc_id="a", title="A", text="a0|a1|a2"),
        ingest.Document(doc_id="b", title="B", text="b0|b1"),
    ])

    await ingest.ingest_documents([ingest.Document(doc_id="a", title="A", text="a0")])

    # a's chunks 1-2 are gone; b wasn't re-ingested and is left alone
    assert await _stored_chunks(qdrant) == [("a", 0), ("b", 0), ("b", 1)]


@pytest.mark.filterwarnings("ignore:Payload indexes have no effect")
async def test_prune_missing_deletes_docs_no_longer_in_the_corpus(qdrant):
    await ingest.ingest_documents([
        ingest.Document(doc_id="a", title="A", text="a0"),
        ingest.Document(doc_id="b", title="B", text="b0|b1"),
    ])

    await ingest.ingest_documents([ingest.Document(doc_id="a", title="A", text="a0")], prune_missing=True)

    assert await _stored_chunks(qdrant) == [("a", 0)]


async def test_store_deletes_refuse_to_match_everything():
    # an empty `should` or MatchAny would select every point in the collection
    await qdrant_store.delete_stale_chunks({})  # no-op; conftest fails any Qdrant call
    with pytest.raises(ValueError):
        await qdrant_store.delete_documents_except([])


async def test_embed_documents_splits_requests_at_the_api_limit(monkeypatch):
    batch_sizes = []

    def fake_embed_sync(texts, task_type):
        batch_sizes.append(len(texts))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[text]) for text in texts])

    monkeypatch.setattr(embeddings, "_embed_sync", fake_embed_sync)
    texts = [f"t{i}" for i in range(2 * embeddings.MAX_TEXTS_PER_REQUEST + 1)]

    vectors = await embeddings.embed_documents(texts)

    assert batch_sizes == [100, 100, 1]
    assert vectors == [[text] for text in texts]  # order preserved across batches


def test_loader_moves_the_h1_heading_into_the_title(tmp_path, monkeypatch):
    (tmp_path / "billing.md").write_text("# Billing and Refunds\n\nBody text.\n")
    (tmp_path / "plain.md").write_text("No heading here.\n")
    monkeypatch.setattr(ingest_script, "DOCS_DIR", tmp_path)

    assert ingest_script.load_documents() == [
        ingest.Document(doc_id="billing", title="Billing and Refunds", text="Body text."),
        ingest.Document(doc_id="plain", title="plain", text="No heading here."),
    ]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (rate_limited_error(quota_id=DAILY), DailyQuotaExhaustedError),
        (rate_limited_error(quota_id=PER_MINUTE), RateLimitedError),
        (ServerError(503, {"error": {"code": 503, "message": "high demand"}}), ModelOverloadedError),
        (ClientError(400, {"error": {"code": 400, "message": "bad"}}), ClientError),
    ],
)
def test_embedding_errors_surface_daily_quota_distinctly(monkeypatch, error, expected):
    def embed_content(**kwargs):
        raise error

    fake_client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
    monkeypatch.setattr(embeddings, "get_gemini_client", lambda: fake_client)
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # the 503 case backs off

    with pytest.raises(expected):
        embeddings._embed_sync(["text"], "RETRIEVAL_QUERY")
