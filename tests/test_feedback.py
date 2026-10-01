"""Visitors' ratings of answers: POST /feedback, the stores, and the export."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

import app.api.feedback as feedback
import app.api.routes as routes
from app.api.feedback import MemoryFeedbackStore, RedisFeedbackStore
from app.core.config import get_settings
from app.main import app
from scripts.feedback_candidates import drafts


def _rating(**overrides):
    return {"answer_id": "a1b2c3d4e5", "rating": "down", "question": "Is the refund window 30 days?",
            "answer": "I don't have enough information.", "sources": ["billing-refunds"]} | overrides


@pytest.fixture
def logged(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(feedback, "log_event", lambda **fields: events.append(fields))
    return events


def test_every_answer_comes_with_an_id_to_rate_it_by(monkeypatch):
    async def run_agent(query, history=None):
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(routes, "run_agent", run_agent)
    monkeypatch.setattr(routes, "log_event", lambda **fields: None)
    client = TestClient(app)

    ids = {client.post("/chat", json={"query": "hi"}).json()["answer_id"] for _ in range(2)}

    assert len(ids) == 2  # one per answer, cached ones included


async def test_a_rating_is_kept_and_logged_without_the_text(logged):
    response = TestClient(app).post("/feedback", json=_rating())

    assert response.status_code == 204
    [kept] = await app.state.feedback.newest()
    assert kept["question"] == "Is the refund window 30 days?" and kept["rating"] == "down"
    assert kept["received_at"].endswith("+00:00")
    [event] = logged
    assert event["event"] == "feedback" and event["rating"] == "down"
    assert "question" not in event and "answer" not in event  # LOG_CHAT_TEXT is off


@pytest.mark.parametrize("bad", [
    {"rating": "meh"},
    {"question": ""},
    {"answer": "x" * 8001},
    {"sources": ["s"] * 11},
    {"answer_id": "x" * 65},
], ids=["rating", "empty-question", "long-answer", "many-sources", "long-id"])
def test_a_malformed_rating_is_refused(logged, bad):
    assert TestClient(app).post("/feedback", json=_rating(**bad)).status_code == 422
    assert logged == []


def test_ratings_have_their_own_rate_limit(logged, monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", None)
    client = TestClient(app)

    codes = [client.post("/feedback", json=_rating()).status_code for _ in range(21)]

    assert codes == [204] * 20 + [429]
    assert client.post("/chat", json={"query": ""}).status_code == 422  # questions are counted separately


# ---- export ------------------------------------------------------------------------------


def test_without_a_token_configured_the_export_does_not_exist():
    assert TestClient(app).get("/feedback/export").status_code == 404


def test_the_export_needs_the_token(monkeypatch, logged):
    monkeypatch.setattr(get_settings(), "feedback_export_token", "s3cret-token")
    client = TestClient(app)
    client.post("/feedback", json=_rating(answer_id="first", rating="up"))
    client.post("/feedback", json=_rating(answer_id="second"))

    assert client.get("/feedback/export").status_code == 401
    assert client.get("/feedback/export", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/feedback/export", headers={"Authorization": "Bearer s3cret-token"})

    assert response.status_code == 200
    assert [r["answer_id"] for r in response.json()["feedback"]] == ["second", "first"]  # newest first


def test_thumbs_down_become_draft_eval_cases_once_per_question():
    ratings = [_rating(answer_id="aaaaaaaa11"), _rating(answer_id="bbbbbbbb22", question="is the refund window 30 days?"),
               _rating(answer_id="cccccccc33", rating="up", question="How do I reset my password?")]
    for rating in ratings:
        rating["received_at"] = "2026-10-01T12:00:00+00:00"

    [draft] = drafts(ratings)

    assert draft["id"] == "feedback_aaaaaaaa" and draft["query"] == "Is the refund window 30 days?"
    assert draft["category"] == "TODO" and draft["feedback"]["bad_answer"] == "I don't have enough information."


# ---- the stores --------------------------------------------------------------------------


@pytest.fixture(params=["memory", "redis"])
def store(request):
    memory = MemoryFeedbackStore()
    if request.param == "memory":
        return memory
    return RedisFeedbackStore(memory, fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer()))


async def test_a_store_keeps_the_newest_ratings_first(store):
    for answer_id in ("1", "2", "3"):
        await store.add({"answer_id": answer_id})

    assert [r["answer_id"] for r in await store.newest()] == ["3", "2", "1"]


async def test_only_the_newest_ratings_are_kept(monkeypatch):
    monkeypatch.setattr(feedback, "MAX_KEPT", 3)
    redis = RedisFeedbackStore(MemoryFeedbackStore(), fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer()))
    memory = MemoryFeedbackStore(max_kept=3)
    for store in (redis, memory):
        for answer_id in range(5):
            await store.add({"answer_id": answer_id})
        assert [r["answer_id"] for r in await store.newest()] == [4, 3, 2]


async def test_while_redis_is_down_ratings_are_kept_in_the_process(logged):
    server = fakeredis.FakeServer()
    server.connected = False
    store = RedisFeedbackStore(MemoryFeedbackStore(), fakeredis.FakeAsyncRedis(server=server))

    await store.add({"answer_id": "x"})

    assert [r["answer_id"] for r in await store.newest()] == ["x"]
    assert [e["event"] for e in logged] == ["redis_unavailable"]
