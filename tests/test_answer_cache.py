"""Reusing answers to repeat first questions: through /chat, and the stores."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

import app.api.answer_cache as answer_cache
import app.api.routes as routes
from app.api.answer_cache import MemoryAnswerCache, RedisAnswerCache, answer_cache_key, build_answer_cache
from app.core.config import get_settings
from app.core.observability import time_step
from app.main import app
from tests.fakes import chunk


@pytest.fixture
def agent(monkeypatch):
    """A fake agent that spends tokens, and records each question it gets."""
    asked: list[str] = []

    async def fake_run_agent(query, history=None):
        asked.append(query)
        with time_step("generate") as usage:
            usage.update(input_tokens=400, output_tokens=60, cost_usd=0.0001)
        return {"answer": f"answer {len(asked)}", "sources": ["billing-refunds", "billing-refunds"],
                "chunks": [chunk("billing-refunds"), chunk("billing-refunds")],
                "router_prompt_version": "router_v3", "answer_prompt_version": "grounded_answer_v4"}

    monkeypatch.setattr(routes, "run_agent", fake_run_agent)
    return asked


@pytest.fixture
def logged(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(routes, "log_event", lambda **fields: events.append(fields))
    return events


def _ask(client, query, session_id=None):
    return client.post("/chat", json={"query": query} | ({"session_id": session_id} if session_id else {})).json()


# ---- through /chat -----------------------------------------------------------------------


def test_a_repeat_first_question_is_answered_from_the_cache_without_the_model(agent, logged):
    client = TestClient(app)
    first = _ask(client, "How do refunds work?")

    again = _ask(client, "  how do REFUNDS work ")  # same question, typed differently

    assert agent == ["How do refunds work?"]
    assert (again["answer"], again["sources"]) == (first["answer"], ["billing-refunds"])
    assert again["session_id"] != first["session_id"]  # still its own conversation
    assert [e["cache_hit"] for e in logged] == [False, True]
    assert (logged[1]["total_tokens"], logged[1]["answer_prompt_version"]) == (0, "grounded_answer_v4")
    # still recorded as an answer from a search; nothing retrieved this time, so null, not 0
    assert logged[1]["used_tool"] is True
    assert logged[1]["num_chunks_retrieved"] is None and logged[1]["retrieval_scores"] is None
    assert logged[0]["num_chunks_retrieved"] == 2  # the first, real run


def test_a_follow_up_is_never_answered_from_the_cache(agent):
    client = TestClient(app)
    _ask(client, "How do refunds work?")
    session = _ask(client, "What about annual plans?")["session_id"]

    _ask(client, "How do refunds work?", session_id=session)  # same text, but mid-conversation

    assert agent == ["How do refunds work?", "What about annual plans?", "How do refunds work?"]


def test_a_cached_answer_still_becomes_part_of_its_conversation(agent, monkeypatch):
    client = TestClient(app)
    _ask(client, "How do refunds work?")
    session = _ask(client, "How do refunds work?")["session_id"]  # from the cache
    seen = []

    async def follow_up(query, history=None):
        seen.extend(history)
        return {"answer": "14 days.", "sources": []}

    monkeypatch.setattr(routes, "run_agent", follow_up)
    _ask(client, "What about annual plans?", session_id=session)

    assert [t.text for t in seen] == ["How do refunds work?", "answer 1"]


def test_a_failed_answer_is_not_cached(agent, monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)
    working = routes.run_agent

    async def failing(query, history=None):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(routes, "run_agent", failing)
    assert client.post("/chat", json={"query": "How do refunds work?"}).status_code == 500
    monkeypatch.setattr(routes, "run_agent", working)

    assert _ask(client, "How do refunds work?")["answer"] == "answer 1"
    assert agent == ["How do refunds work?"]  # asked for real, not served a cached failure


def test_changing_what_shapes_an_answer_misses_instead_of_serving_the_old_one(monkeypatch):
    before = answer_cache_key("How do refunds work?")
    version = answer_cache.GROUNDED_ANSWER_PROMPT_VERSION

    monkeypatch.setattr(answer_cache, "GROUNDED_ANSWER_PROMPT_VERSION", "grounded_answer_v99")
    after_prompt_change = answer_cache_key("How do refunds work?")
    monkeypatch.setattr(answer_cache, "GROUNDED_ANSWER_PROMPT_VERSION", version)
    monkeypatch.setattr(get_settings(), "retrieval_max_score_gap", get_settings().retrieval_max_score_gap + 0.05)
    after_gap_change = answer_cache_key("How do refunds work?")

    assert len({before, after_prompt_change, after_gap_change}) == 3


def test_the_key_never_contains_the_question():
    assert "refund" not in answer_cache_key("How do refunds work?")


def test_a_ttl_of_zero_turns_caching_off(monkeypatch):
    monkeypatch.setattr(get_settings(), "answer_cache_ttl_s", 0)
    assert build_answer_cache(get_settings()) is None


# ---- the stores --------------------------------------------------------------------------


@pytest.fixture(params=["memory", "redis"])
def store(request):
    memory = MemoryAnswerCache(ttl_s=3600)
    if request.param == "memory":
        return memory
    return RedisAnswerCache(memory, fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer()))


async def test_a_store_returns_what_was_put(store):
    await store.put("k", {"answer": "a", "sources": ["billing-refunds"]})

    assert await store.get("k") == {"answer": "a", "sources": ["billing-refunds"]}
    assert await store.get("other") is None


async def test_a_memory_answer_expires_after_its_ttl():
    now = [0.0]
    store = MemoryAnswerCache(ttl_s=3600, clock=lambda: now[0])
    await store.put("k", {"answer": "a"})

    now[0] = 3599.0
    assert await store.get("k") is not None
    now[0] = 3600.0
    assert await store.get("k") is None


async def test_a_redis_answer_is_stored_with_its_ttl_and_survives_a_restart():
    server = fakeredis.FakeServer()
    redis = fakeredis.FakeAsyncRedis(server=server)
    await RedisAnswerCache(MemoryAnswerCache(3600), redis).put("k", {"answer": "a"})

    restarted = RedisAnswerCache(MemoryAnswerCache(3600), fakeredis.FakeAsyncRedis(server=server))
    assert await restarted.get("k") == {"answer": "a"}
    assert 0 < await redis.ttl("answer:k") <= 3600


async def test_while_redis_is_down_answers_are_cached_in_the_process(monkeypatch):
    events = []
    monkeypatch.setattr(answer_cache, "log_event", lambda **fields: events.append(fields))
    server = fakeredis.FakeServer()
    server.connected = False
    store = RedisAnswerCache(MemoryAnswerCache(3600), fakeredis.FakeAsyncRedis(server=server))

    await store.put("k", {"answer": "a"})

    assert await store.get("k") == {"answer": "a"}
    assert [e["event"] for e in events] == ["redis_unavailable"]  # logged once, not per call


async def test_a_store_can_delete_an_answer(store):
    await store.put("k", {"answer": "a"})

    await store.delete("k")

    assert await store.get("k") is None
    await store.delete("never-there")  # deleting a missing key is fine
