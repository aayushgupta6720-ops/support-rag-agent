import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.core.config import get_settings
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    ModelConnectionError,
    ModelOutputError,
    ModelOverloadedError,
    ModelServerError,
    ModelTimeoutError,
    RateLimitedError,
)
from app.rag.qdrant_store import SearchUnavailableError
from app.core.observability import time_step
from app.main import app
from tests.fakes import chunk


@pytest.fixture
def logged(monkeypatch):
    events: list[dict] = []
    monkeypatch.setattr(routes, "log_event", lambda **fields: events.append(fields))
    return events


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_returns_answer_and_logs_one_structured_event(client, logged, monkeypatch):
    async def fake_run_agent(query, history=None):
        with time_step("generate") as usage:
            usage.update(input_tokens=30, output_tokens=5, cost_usd=0.001)
        return {
            "answer": "Use the Forgot password link.",
            "sources": ["password-reset"],
            "chunks": [chunk("password-reset", score=0.71234)],
            "router_prompt_version": "router_v1",
            "answer_prompt_version": "grounded_answer_v2",
        }

    monkeypatch.setattr(routes, "run_agent", fake_run_agent)

    response = client.post("/chat", json={"query": "reset password", "session_id": "s1"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use the Forgot password link."
    assert body["sources"] == ["password-reset"]
    assert body["latency_ms"] >= 0

    [event] = logged
    assert event["event"] == "chat_call"
    assert event["session_id"] == "s1"
    assert event["used_tool"] is True
    assert event["retrieval_scores"] == [0.7123]
    assert event["answer_prompt_version"] == "grounded_answer_v2"
    assert event["total_tokens"] == 35
    assert [s["name"] for s in event["steps"]] == ["generate"]


def test_chat_lists_each_source_doc_once(client, logged, monkeypatch):
    # sources come one per retrieved chunk, and a doc often has two of the top 4
    per_chunk = ["password-reset", "two-factor-auth", "two-factor-auth", "billing-refunds"]

    async def fake_run_agent(query, history=None):
        return {"answer": "...", "sources": per_chunk, "chunks": [chunk(doc_id) for doc_id in per_chunk]}

    monkeypatch.setattr(routes, "run_agent", fake_run_agent)

    response = client.post("/chat", json={"query": "reset password"})

    assert response.json()["sources"] == ["password-reset", "two-factor-auth", "billing-refunds"]
    # the log keeps one entry per chunk, lined up with retrieval_scores
    assert logged[0]["sources"] == per_chunk


def test_chat_rejects_empty_query(client, logged):
    assert client.post("/chat", json={"query": ""}).status_code == 422
    assert logged == []


def test_chat_failure_is_logged_with_error_then_returns_500(client, logged, monkeypatch):
    async def failing_agent(query, history=None):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(routes, "run_agent", failing_agent)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 500
    [event] = logged
    assert event["error"] == "gemini exploded"


def test_daily_quota_returns_503_with_a_clear_message(client, logged, monkeypatch):
    async def quota_exhausted(query, history=None):
        raise DailyQuotaExhaustedError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(routes, "run_agent", quota_exhausted)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 503
    body = response.json()
    assert "daily quota" in body["detail"]
    assert "midnight Pacific time (in about" in body["detail"]
    assert "RESOURCE_EXHAUSTED" not in body["detail"]
    assert body["resets_at"].startswith(("20", "21"))  # an ISO timestamp
    assert 0 < int(response.headers["Retry-After"]) <= 24 * 3600
    [event] = logged
    assert event["error_type"] == "DailyQuotaExhaustedError"


def test_per_minute_quota_past_the_retries_returns_429_not_500(client, logged, monkeypatch):
    async def rate_limited(query, history=None):
        raise RateLimitedError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(routes, "run_agent", rate_limited)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 429
    assert "Wait a minute" in response.json()["detail"]
    assert response.headers["Retry-After"] == "60"


def test_overloaded_model_returns_503_saying_it_is_temporary(client, logged, monkeypatch):
    async def overloaded(query, history=None):
        raise ModelOverloadedError("503 UNAVAILABLE. {'error': {...}}")

    monkeypatch.setattr(routes, "run_agent", overloaded)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 503
    assert "Gemini is overloaded right now" in response.json()["detail"]
    assert "UNAVAILABLE" not in response.json()["detail"]
    assert response.headers["Retry-After"] == "60"
    [event] = logged
    assert event["error_type"] == "ModelOverloadedError"


def test_gemini_internal_error_returns_503_saying_it_is_temporary(client, logged, monkeypatch):
    async def internal_error(query, history=None):
        raise ModelServerError("500 INTERNAL. {'error': {...}}")

    monkeypatch.setattr(routes, "run_agent", internal_error)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 503
    assert "Gemini had an internal error" in response.json()["detail"]
    assert "INTERNAL" not in response.json()["detail"]
    assert response.headers["Retry-After"] == "60"
    [event] = logged
    assert event["error_type"] == "ModelServerError"


def test_a_failed_gemini_connection_returns_503_saying_it_is_temporary(client, logged, monkeypatch):
    async def disconnected(query, history=None):
        raise ModelConnectionError("Server disconnected without sending a response.")

    monkeypatch.setattr(routes, "run_agent", disconnected)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 503
    assert "The connection to Gemini failed" in response.json()["detail"]
    assert response.headers["Retry-After"] == "60"
    [event] = logged
    assert event["error_type"] == "ModelConnectionError"


def test_gemini_timeout_returns_504_saying_it_was_stopped(client, logged, monkeypatch):
    async def timed_out(query, history=None):
        raise ModelTimeoutError("no response from Gemini within 60s")

    monkeypatch.setattr(routes, "run_agent", timed_out)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 504
    assert "didn't respond within 60 seconds" in response.json()["detail"]
    [event] = logged
    assert event["error_type"] == "ModelTimeoutError"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(8 * 3600 - 300, "in about 8 hours"), (3600, "in about 1 hour"), (90, "in about 2 minutes"), (5, "in about 1 minute")],
)
def test_reset_time_reads_naturally(seconds, expected):
    assert routes._in_about(seconds) == expected


def _failing_with(exc):
    async def run_agent(query, history=None):
        raise exc
    return run_agent


def test_a_search_outage_returns_503_saying_it_is_temporary(client, logged, monkeypatch):
    monkeypatch.setattr(routes, "run_agent", _failing_with(SearchUnavailableError("ResponseHandlingException: timed out")))

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 503 and response.headers["Retry-After"] == "30"
    assert "help articles failed" in response.json()["detail"]
    assert "ResponseHandlingException" not in response.json()["detail"]
    assert logged[0]["error_type"] == "SearchUnavailableError"


def test_a_blocked_or_unparseable_answer_returns_502_asking_to_rephrase(client, logged, monkeypatch):
    monkeypatch.setattr(routes, "run_agent", _failing_with(ModelOutputError("prompt blocked: SAFETY")))

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 502
    assert "rephras" in response.json()["detail"]


def _answering(search_query="reset link expiry"):
    async def run_agent(query, history=None):
        return {"answer": "30 minutes.", "sources": [], "search_query": search_query}
    return run_agent


def test_the_log_line_leaves_out_the_question_by_default(client, logged, monkeypatch):
    monkeypatch.setattr(routes, "run_agent", _answering())

    client.post("/chat", json={"query": "my email is me@example.com, reset link?"})

    [event] = logged
    assert "query" not in event and "search_query" not in event
    assert event["query_chars"] == len("my email is me@example.com, reset link?")


def test_log_chat_text_puts_the_question_and_search_query_in_the_log(client, logged, monkeypatch):
    monkeypatch.setattr(get_settings(), "log_chat_text", True)
    monkeypatch.setattr(routes, "run_agent", _answering())

    client.post("/chat", json={"query": "how long is the reset link good for?"})

    [event] = logged
    assert event["query"] == "how long is the reset link good for?"
    assert event["search_query"] == "reset link expiry"


def test_surrounding_whitespace_is_stripped_before_the_agent_sees_the_question(client, logged, monkeypatch):
    seen = []

    async def run_agent(query, history=None):
        seen.append(query)
        return {"answer": "ok", "sources": []}

    monkeypatch.setattr(routes, "run_agent", run_agent)

    assert client.post("/chat", json={"query": "  reset password \n"}).status_code == 200
    assert client.post("/chat", json={"query": " \n\t "}).status_code == 422
    assert seen == ["reset password"]
