"""Multi-turn chat: session history through /chat, and the stores behind it."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
import app.api.sessions as sessions
from app.agent.history import Turn
from app.api.sessions import MemorySessionStore, RedisSessionStore
from app.core.config import get_settings
from app.main import app


@pytest.fixture
def agent(monkeypatch):
    """A fake agent that answers "answer N" and records the history it got."""
    seen: list[list[Turn]] = []

    async def fake_run_agent(query, history=None):
        seen.append(list(history or []))
        return {"answer": f"answer {len(seen)}", "sources": []}

    monkeypatch.setattr(routes, "run_agent", fake_run_agent)
    monkeypatch.setattr(routes, "log_event", lambda **fields: None)
    monkeypatch.setattr(app.state, "answer_cache", None)  # these count every request reaching the agent
    return seen


def _ask(client, query, session_id=None, visitor="198.51.100.4"):
    body = {"query": query} | ({"session_id": session_id} if session_id else {})
    return client.post("/chat", json=body, headers={"CF-Connecting-IP": visitor})


@pytest.fixture
def behind_render(monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")


# ---- through /chat -----------------------------------------------------------------------


def test_a_follow_up_gets_the_earlier_exchange(agent, behind_render):
    client = TestClient(app)
    first = _ask(client, "How do I get a refund?").json()

    second = _ask(client, "What about annual plans?", session_id=first["session_id"]).json()

    assert agent == [[], [Turn("user", "How do I get a refund?"), Turn("model", "answer 1")]]
    assert second["session_id"] == first["session_id"]


def test_each_new_conversation_gets_its_own_unguessable_id(agent, behind_render):
    client = TestClient(app)
    ids = {_ask(client, "hi").json()["session_id"] for _ in range(3)}
    assert len(ids) == 3 and all(len(i) >= 20 for i in ids)
    assert agent == [[], [], []]


def test_a_session_id_only_continues_the_same_visitors_conversation(agent, behind_render):
    # ids can be chosen by the client, so another visitor sending the same one
    # must not see (or be able to ask the model to repeat) this conversation
    client = TestClient(app)
    _ask(client, "my email is me@example.com", session_id="team-chat", visitor="198.51.100.4")

    _ask(client, "what did I just say?", session_id="team-chat", visitor="203.0.113.9")

    assert agent[1] == []


def test_a_failed_answer_is_not_added_to_the_history(agent, behind_render, monkeypatch):
    client = TestClient(app, raise_server_exceptions=False)

    async def failing(query, history=None):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(routes, "run_agent", failing)
    assert _ask(client, "first", session_id="s1").status_code == 500

    async def recording(query, history=None):
        agent.append(list(history or []))
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(routes, "run_agent", recording)
    _ask(client, "second", session_id="s1")

    assert agent == [[]]


def test_history_keeps_only_the_last_few_exchanges(agent, behind_render, monkeypatch):
    monkeypatch.setattr(app.state, "sessions", MemorySessionStore(max_exchanges=2, ttl_s=1800))
    client = TestClient(app)
    for i in range(4):
        _ask(client, f"q{i}", session_id="s1")

    assert [t.text for t in agent[-1]] == ["q1", "answer 2", "q2", "answer 3"]


# ---- the stores --------------------------------------------------------------------------


@pytest.fixture(params=["memory", "redis"])
def store(request):
    memory = MemorySessionStore(max_exchanges=2, ttl_s=1800)
    if request.param == "memory":
        return memory
    return RedisSessionStore(memory, fakeredis.FakeAsyncRedis(server=fakeredis.FakeServer()))


async def test_a_store_keeps_exchanges_in_order_and_trims_the_oldest(store):
    for i in range(3):
        await store.add_exchange("v|s", f"q{i}", f"a{i}")

    assert await store.history("v|s") == [Turn("user", "q1"), Turn("model", "a1"), Turn("user", "q2"), Turn("model", "a2")]
    assert await store.history("v|other") == []


async def test_a_memory_session_ends_once_idle_for_its_ttl():
    now = [0.0]
    store = MemorySessionStore(max_exchanges=3, ttl_s=1800, clock=lambda: now[0])
    await store.add_exchange("v|s", "q", "a")

    now[0] = 1799.0
    assert await store.history("v|s") != []
    await store.add_exchange("v|s", "q2", "a2")  # activity pushes the expiry back
    now[0] = 1799.0 + 1800.0
    assert await store.history("v|s") == []


async def test_redis_sessions_expire_and_survive_a_restart():
    server = fakeredis.FakeServer()
    redis = fakeredis.FakeAsyncRedis(server=server)
    await RedisSessionStore(MemorySessionStore(3, 1800), redis).add_exchange("v|s", "q", "a")

    restarted = RedisSessionStore(MemorySessionStore(3, 1800), fakeredis.FakeAsyncRedis(server=server))
    assert await restarted.history("v|s") == [Turn("user", "q"), Turn("model", "a")]
    assert 0 < await redis.ttl("session:v|s") <= 1800


async def test_while_redis_is_down_sessions_are_kept_in_the_process(monkeypatch):
    events = []
    monkeypatch.setattr(sessions, "log_event", lambda **fields: events.append(fields))
    server = fakeredis.FakeServer()
    server.connected = False
    store = RedisSessionStore(MemorySessionStore(3, 1800), fakeredis.FakeAsyncRedis(server=server))

    await store.add_exchange("v|s", "q", "a")

    assert await store.history("v|s") == [Turn("user", "q"), Turn("model", "a")]
    assert [e["event"] for e in events] == ["redis_unavailable"]  # logged once, not per call


def test_each_answer_says_how_many_earlier_messages_it_could_see(agent, behind_render, monkeypatch):
    # the widget compares this with its own transcript to spot an expired session
    now = [0.0]
    monkeypatch.setattr(app.state, "sessions", MemorySessionStore(max_exchanges=3, ttl_s=1800, clock=lambda: now[0]))
    client = TestClient(app)
    first = _ask(client, "How do I get a refund?").json()
    assert first["history_turns"] == 0

    assert _ask(client, "What about annual plans?", session_id=first["session_id"]).json()["history_turns"] == 2

    now[0] = 1800.0 + 1  # idle past the TTL
    assert _ask(client, "And monthly?", session_id=first["session_id"]).json()["history_turns"] == 0
