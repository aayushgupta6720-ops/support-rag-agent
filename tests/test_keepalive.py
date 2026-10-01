"""The search check behind the keep-alive workflow: /health/search."""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

import app.api.routes as routes
import app.rag.qdrant_store as qdrant_store
from app.core.config import get_settings
from app.main import app
from app.rag.qdrant_store import SearchUnavailableError


@pytest.fixture
def logged(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(routes, "log_event", lambda **fields: events.append(fields))
    return events


def _checking(result):
    async def check_search():
        if isinstance(result, Exception):
            raise result
        return result
    return check_search


def test_a_working_search_is_healthy_and_logged(monkeypatch, logged):
    monkeypatch.setattr(routes, "check_search", _checking(14))

    response = TestClient(app).get("/health/search")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy" and response.json()["points"] == 14
    assert logged[0]["event"] == "search_check" and logged[0]["ok"] is True


def test_a_failed_search_is_a_503_so_the_workflow_fails(monkeypatch, logged):
    monkeypatch.setattr(routes, "check_search", _checking(SearchUnavailableError("ResponseHandlingException: timed out")))

    response = TestClient(app).get("/health/search")

    assert response.status_code == 503 and "help articles failed" in response.json()["detail"]
    assert logged[0]["ok"] is False


def test_an_empty_index_is_a_503_too(monkeypatch, logged):
    # reachable but never ingested (say, a recreated cluster): every question would find nothing
    monkeypatch.setattr(routes, "check_search", _checking(0))

    response = TestClient(app).get("/health/search")

    assert response.status_code == 503 and response.json()["detail"] == "The help-article index is empty."


def test_search_checks_are_rate_limited(monkeypatch, logged):
    monkeypatch.setattr(routes, "check_search", _checking(14))
    client = TestClient(app)

    assert [client.get("/health/search").status_code for _ in range(7)] == [200] * 6 + [429]
    assert client.get("/health").status_code == 200  # the platform's health check is unaffected


async def test_check_search_runs_a_real_search_and_counts_the_points(monkeypatch):
    client = AsyncQdrantClient(location=":memory:")
    monkeypatch.setattr(qdrant_store, "get_client", lambda: client)
    settings = get_settings()
    await client.create_collection(settings.qdrant_collection,
                                   vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE))
    await client.upsert(settings.qdrant_collection, points=[
        PointStruct(id=i, vector=[0.1 * (i + 1)] * settings.embedding_dim, payload={"doc_id": "d"}) for i in range(3)])

    assert await qdrant_store.check_search() == 3


async def test_a_failing_count_is_reported_as_search_unavailable(monkeypatch):
    class SearchesButCannotCount:
        async def query_points(self, **kwargs):
            return type("Response", (), {"points": []})()

        async def count(self, **kwargs):
            raise httpx.ConnectError("connection reset")

    monkeypatch.setattr(qdrant_store, "get_client", lambda: SearchesButCannotCount())

    with pytest.raises(SearchUnavailableError):
        await qdrant_store.check_search()


def test_the_workflow_calls_the_search_check_twice_a_week():
    workflow = (Path(__file__).resolve().parent.parent / ".github" / "workflows" / "keepalive.yml").read_text()
    assert "/health/search" in workflow
    assert 'cron: "17 6 * * 1,4"' in workflow  # under Qdrant's 7-day limit even if a run is delayed
