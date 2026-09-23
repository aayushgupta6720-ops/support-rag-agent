import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes
from app.core.gemini_client import DailyQuotaExhaustedError
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
    async def fake_run_agent(query):
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


def test_chat_rejects_empty_query(client, logged):
    assert client.post("/chat", json={"query": ""}).status_code == 422
    assert logged == []


def test_chat_failure_is_logged_with_error_then_returns_500(client, logged, monkeypatch):
    async def failing_agent(query):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(routes, "run_agent", failing_agent)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 500
    [event] = logged
    assert event["error"] == "gemini exploded"


def test_daily_quota_returns_503_with_a_clear_message(client, logged, monkeypatch):
    async def quota_exhausted(query):
        raise DailyQuotaExhaustedError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(routes, "run_agent", quota_exhausted)

    response = client.post("/chat", json={"query": "anything"})

    assert response.status_code == 503
    assert "daily quota" in response.json()["detail"]
    [event] = logged
    assert event["error_type"] == "DailyQuotaExhaustedError"
