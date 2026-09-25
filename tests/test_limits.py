"""Per-visitor rate limit, query length cap and request body cap on /chat."""

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import app.api.ratelimit as ratelimit
import app.api.routes as routes
from app.api.body_limit import BodySizeLimit
from app.api.ratelimit import Limit, RateLimiter, build_rate_limiters, client_key
from app.core.config import get_settings
from app.main import app


@pytest.fixture
def agent_calls(monkeypatch):
    calls: list[str] = []

    async def fake_run_agent(query):
        calls.append(query)
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(routes, "run_agent", fake_run_agent)
    monkeypatch.setattr(routes, "log_event", lambda **fields: None)
    return calls


@pytest.fixture
def behind_render(monkeypatch):
    """The live setup: visitors identified by Cloudflare's header."""
    settings = get_settings()
    monkeypatch.setattr(settings, "client_ip_header", "CF-Connecting-IP")
    monkeypatch.setattr(app.state, "rate_limiters", build_rate_limiters(settings))


def _ask(client, visitor, **headers):
    return client.post("/chat", json={"query": "reset password"}, headers={"CF-Connecting-IP": visitor, **headers})


# ---- rate limit ------------------------------------------------------------------------


def test_a_visitor_over_the_limit_gets_a_429_and_the_agent_is_not_run(agent_calls, behind_render):
    client = TestClient(app)
    assert [_ask(client, "198.51.100.4").status_code for _ in range(6)] == [200] * 6

    refused = _ask(client, "198.51.100.4")

    assert refused.status_code == 429 and len(agent_calls) == 6
    assert refused.headers["Retry-After"] == "60"
    assert refused.json()["detail"] == "You've reached the limit of 6 questions a minute. Try again in 60 seconds."


def test_forged_forwarded_headers_do_not_reset_the_count_and_others_are_unaffected(agent_calls, behind_render):
    client = TestClient(app)
    for _ in range(6):
        _ask(client, "198.51.100.4")

    assert _ask(client, "198.51.100.4", **{"X-Forwarded-For": "1.2.3.4"}).status_code == 429
    assert _ask(client, "203.0.113.9").status_code == 200


def _request(peer, **headers):
    raw = [(k.replace("_", "-").lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "headers": raw, "client": (peer, 1234)})


def test_without_a_configured_header_the_connecting_address_is_used_and_forwarded_ones_ignored(monkeypatch):
    monkeypatch.setattr(get_settings(), "client_ip_header", None)
    assert client_key(_request("203.0.113.7", x_forwarded_for="1.1.1.1", cf_connecting_ip="2.2.2.2")) == "203.0.113.7"


def test_a_missing_proxy_header_puts_everyone_in_one_count(monkeypatch):
    # not the connecting address, which uvicorn may have taken from X-Forwarded-For
    monkeypatch.setattr(get_settings(), "client_ip_header", "CF-Connecting-IP")
    monkeypatch.setattr(ratelimit, "log_event", lambda **fields: None)
    assert {client_key(_request(peer)) for peer in ("10.9.9.1", "10.9.9.2")} == {"missing CF-Connecting-IP"}


def test_the_window_slides_and_refused_requests_do_not_count():
    now = [1000.0]
    limiter = RateLimiter("questions", [Limit(2, 60, "a minute")], clock=lambda: now[0])
    assert limiter.hit("a") is None and limiter.hit("a") is None
    for _ in range(10):
        now[0] += 5
        assert limiter.hit("a") is not None
    now[0] += 10  # 60s after the first request
    assert limiter.hit("a") is None


# ---- query length ------------------------------------------------------------------------


def test_queries_and_session_ids_have_a_length_cap(agent_calls):
    client = TestClient(app)
    assert client.post("/chat", json={"query": "x" * 2000}).status_code == 200
    assert client.post("/chat", json={"query": "x" * 2001}).status_code == 422
    assert client.post("/chat", json={"query": "hi", "session_id": "s" * 65}).status_code == 422
    assert agent_calls == ["x" * 2000]


# ---- body size -------------------------------------------------------------------------


def test_an_oversized_body_is_refused_before_the_app_sees_it(agent_calls):
    response = TestClient(app).post(
        "/chat", content=b'{"query": "' + b"x" * (64 * 1024) + b'"}', headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "The request is over the 64 KB limit."}
    assert agent_calls == []


async def _run(middleware, headers, chunks):
    """Drive the middleware with a body sent in chunks; returns (status, what the app read)."""
    incoming = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]
    reads, sent = [], []

    async def receive():
        return incoming.pop(0) if incoming else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    async def inner_app(scope, receive, send):
        while True:
            message = await receive()
            reads.append(message.get("body", b""))
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    await BodySizeLimit(inner_app, **middleware)(
        {"type": "http", "path": "/chat", "headers": headers}, receive, send
    )
    return sent[0]["status"], b"".join(reads)


async def test_a_declared_length_over_the_limit_is_refused_without_reading_the_body():
    status, read = await _run({"max_bytes": 1000}, [(b"content-length", b"100000000")], [])
    assert status == 413 and read == b""


async def test_a_chunked_body_is_cut_off_once_it_passes_the_limit():
    status, read = await _run({"max_bytes": 1000}, [], [b"x" * 600, b"x" * 600, b"x" * 600])
    assert status == 413 and read == b""


async def test_a_body_under_the_limit_reaches_the_app_intact_and_paths_can_have_their_own_limit():
    assert await _run({"max_bytes": 1000}, [], [b"a" * 600, b"b" * 300]) == (200, b"a" * 600 + b"b" * 300)
    assert (await _run({"max_bytes": 10, "max_bytes_by_path": {"/chat": 1000}}, [], [b"x" * 900]))[0] == 200


def test_an_invalid_question_still_counts_so_the_readme_check_costs_no_quota(agent_calls, behind_render):
    # The README's post-deploy check relies on this: 422s that never reach the model, then a 429.
    client = TestClient(app)
    codes = [client.post("/chat", json={"query": ""}, headers={"CF-Connecting-IP": "198.51.100.4"}).status_code
             for _ in range(7)]
    assert codes == [422] * 6 + [429] and agent_calls == []
