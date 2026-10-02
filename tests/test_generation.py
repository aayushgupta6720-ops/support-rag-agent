import contextvars
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import types
from google.genai.errors import ClientError, ServerError

import app.api.routes as routes
import app.core.gemini_client as gemini_client
import app.core.generation as generation
import app.rag.embeddings as embeddings
import app.rag.retrieval as retrieval
from app.agent.graph import AgentAnswer
from app.agent.tools import SEARCH_DOCS_TOOL_NAME
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    ModelConnectionError,
    ModelOverloadedError,
    ModelServerError,
    ModelTimeoutError,
    RateLimitedError,
    is_daily_quota_error,
    next_daily_quota_reset,
)
from app.main import app
from tests.fakes import DAILY, PER_MINUTE, model_response, rate_limited_error


class FakeModels:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def client(monkeypatch):
    """Returns a setter: client(outcomes) installs a fake Gemini client and
    records sleeps instead of actually waiting."""
    sleeps: list[float] = []
    monkeypatch.setattr(generation.time, "sleep", sleeps.append)

    def install(outcomes):
        models = FakeModels(outcomes)
        monkeypatch.setattr(generation, "get_gemini_client", lambda: SimpleNamespace(models=models))
        return models, sleeps

    return install


def _call():
    return generation._generate_sync(contents=[], system_instruction="sys")


def test_retry_delay_parsed_from_retry_info():
    assert generation._rate_limit_retry_delay(rate_limited_error("7s"), fallback=1) == 7.0
    assert generation._rate_limit_retry_delay(rate_limited_error("0.5s"), fallback=1) == 0.5


def test_retry_delay_falls_back_without_retry_info():
    assert generation._rate_limit_retry_delay(rate_limited_error(None), fallback=4) == 4


def test_retries_rate_limit_then_succeeds(client):
    models, sleeps = client([rate_limited_error("3s"), rate_limited_error(None), "response"])

    assert _call() == "response"
    assert models.calls == 3
    # server-suggested delay first, then exponential fallback (2**attempt)
    assert sleeps == [3.0, 4]


def test_gives_up_after_max_retries(client):
    models, sleeps = client([rate_limited_error()] * generation._MAX_RATE_LIMIT_RETRIES)

    # distinct from other ClientErrors, so /chat can say "wait a minute"
    with pytest.raises(RateLimitedError):
        _call()
    assert models.calls == generation._MAX_RATE_LIMIT_RETRIES
    assert len(sleeps) == generation._MAX_RATE_LIMIT_RETRIES - 1


@pytest.mark.parametrize(
    "error",
    [
        ClientError(400, {"error": {"code": 400, "message": "bad request"}}),
        ServerError(501, {"error": {"code": 501, "message": "not implemented"}}),
    ],
)
def test_other_errors_are_not_retried(client, error):
    models, sleeps = client([error, "never reached"])

    with pytest.raises(type(error)):
        _call()
    assert models.calls == 1
    assert sleeps == []


def test_daily_quota_fails_fast_instead_of_sleeping_on_retry_delay(client):
    # Gemini still sends retryDelay ~59s on a per-day quota; trusting it made
    # every /chat call hang for minutes and then 500 once the quota ran out.
    models, sleeps = client([rate_limited_error("59s", quota_id=DAILY), "never reached"])

    with pytest.raises(DailyQuotaExhaustedError):
        _call()
    assert models.calls == 1
    assert sleeps == []


def test_per_minute_quota_is_still_retried(client):
    models, sleeps = client([rate_limited_error("5s", quota_id=PER_MINUTE), "response"])

    assert _call() == "response"
    assert sleeps == [5.0]


def test_only_429s_naming_a_per_day_quota_count_as_daily():
    assert is_daily_quota_error(rate_limited_error(quota_id=DAILY))
    assert not is_daily_quota_error(rate_limited_error(quota_id=PER_MINUTE))
    assert not is_daily_quota_error(rate_limited_error(quota_id=None))
    assert not is_daily_quota_error(ClientError(400, {"error": {"code": 400, "message": "bad"}}))


@pytest.mark.parametrize(
    ("now_utc", "expected"),
    [
        # 16:06 PDT on the 23rd: resets at the coming midnight
        ("2026-09-23T23:06:00+00:00", "2026-09-24T00:00:00-07:00"),
        # 00:30 PDT on the 24th: just reset, so the next one is a day away
        ("2026-09-24T07:30:00+00:00", "2026-09-25T00:00:00-07:00"),
        # after DST ends, midnight Pacific is UTC-8
        ("2026-11-05T12:00:00+00:00", "2026-11-06T00:00:00-08:00"),
    ],
)
def test_daily_quota_resets_at_the_next_pacific_midnight(now_utc, expected):
    from datetime import datetime

    assert next_daily_quota_reset(datetime.fromisoformat(now_utc)).isoformat() == expected


def _overloaded() -> ServerError:
    return ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def test_overload_503_is_retried_with_short_backoff_then_succeeds(client):
    models, sleeps = client([_overloaded(), _overloaded(), "response"])

    assert _call() == "response"
    assert models.calls == 3
    assert sleeps == [1, 2]


def test_persistent_overload_becomes_model_overloaded_error(client):
    models, sleeps = client([_overloaded()] * 4)

    with pytest.raises(ModelOverloadedError):
        _call()
    assert models.calls == 4
    assert sleeps == [1, 2, 4]  # ~7s in all: long enough for a blip, not a spike


def _internal_error() -> ServerError:
    return ServerError(500, {"error": {"code": 500, "message": "An internal error has occurred.", "status": "INTERNAL"}})


def test_internal_500_is_retried_like_an_overload(client):
    models, sleeps = client([_internal_error(), "response"])

    assert _call() == "response"
    assert sleeps == [1]


def test_persistent_500_becomes_model_server_error(client):
    # its own error, so the log tells an internal error from an overload
    models, sleeps = client([_internal_error()] * 4)

    with pytest.raises(ModelServerError):
        _call()
    assert models.calls == 4
    assert sleeps == [1, 2, 4]


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("[Errno 61] Connection refused"),
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadError("[Errno 54] Connection reset by peer"),
    ],
    ids=["unreachable", "disconnected", "reset"],
)
def test_a_dropped_connection_is_retried_like_an_overload(client, error):
    models, sleeps = client([error, "response"])

    assert _call() == "response"
    assert sleeps == [1]


def test_a_connection_that_keeps_failing_becomes_model_connection_error(client):
    models, sleeps = client([httpx.RemoteProtocolError("Server disconnected")] * 4)

    with pytest.raises(ModelConnectionError):
        _call()
    assert models.calls == 4
    assert sleeps == [1, 2, 4]


def test_a_call_that_times_out_fails_fast_without_retrying(client):
    models, sleeps = client([httpx.ReadTimeout("timed out"), "never reached"])

    with pytest.raises(ModelTimeoutError):
        _call()
    assert models.calls == 1
    assert sleeps == []


def test_gemini_client_is_built_with_the_configured_timeout(monkeypatch):
    from app.core import gemini_client
    from app.core.config import Settings

    monkeypatch.setattr(gemini_client, "get_settings", lambda: Settings(_env_file=None, gemini_api_key="x", gemini_timeout_s=12.5))
    gemini_client.get_gemini_client.cache_clear()
    try:
        client = gemini_client.get_gemini_client()
        assert client._api_client._http_options.timeout == 12_500  # milliseconds
    finally:
        gemini_client.get_gemini_client.cache_clear()


# ---- a whole /chat vs the MCP proxy's timeout --------------------------------------------

_MCP_PROXY_TIMEOUT_S = 240  # mcp_server/server.py


class _FakeClock:
    """Stands in for the `time` module in the retry loops: sleeps and fake
    Gemini requests move `now` on instead of waiting."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(gemini_client, "time", clock)
    monkeypatch.setattr(generation, "time", clock)
    return clock


def test_inside_a_retry_window_a_429_is_retried_only_while_the_wait_fits(monkeypatch, clock):
    models = FakeModels([rate_limited_error("7s"), rate_limited_error("7s"), "never reached"])
    monkeypatch.setattr(generation, "get_gemini_client", lambda: SimpleNamespace(models=models))

    def chat():
        gemini_client.start_retry_window(10)
        _call()

    # the first 7s wait fits in 10s; a second one would end at 14s
    with pytest.raises(RateLimitedError):
        contextvars.copy_context().run(chat)  # a context of its own, as each request gets
    assert (models.calls, clock.now) == (2, 7)


def _gemini_method(clock, outcomes):
    """Returns (seconds, result or exception) outcomes in order, moving the
    clock on by each one's seconds."""
    outcomes = list(outcomes)

    def method(**kwargs):
        seconds, outcome = outcomes.pop(0)
        clock.now += seconds
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return method


async def _no_chunks(vector, top_k):
    return []


# Errors come back fast; answers as slowly as the 60s timeout allows.
_429 = (1, rate_limited_error("59s", quota_id=PER_MINUTE))
_503 = (1, _overloaded())
_500 = (1, _internal_error())
_DROPPED = (1, httpx.RemoteProtocolError("Server disconnected"))
_ROUTED = (60, model_response(function_call=(SEARCH_DOCS_TOOL_NAME, {"query": "q"})))
_EMBEDDED = (60, SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0])], metadata=None))
_ANSWERED = (60, model_response(parsed=AgentAnswer(answer="ok")))


@pytest.mark.parametrize(
    ("generate", "embed", "status"),
    [
        # sustained per-minute 429s on the route call, each suggesting a 59s wait
        pytest.param([_429] * 4 + [_ROUTED, _ANSWERED], [_EMBEDDED], 429, id="per-minute-429s"),
        # the loops nest: each 429 retry starts another round of 503 backoff
        pytest.param(([_503] * 3 + [_429]) * 4 + [_503] * 3 + [_ROUTED, _ANSWERED], [_EMBEDDED], 429,
                     id="503s-then-429s"),
        # Gemini's own 500s on the route call, retried and then refused like an overload
        pytest.param([_500] * 4, [_EMBEDDED], 503, id="persistent-500s"),
        # the route call's connection keeps dropping
        pytest.param([_DROPPED] * 4, [_EMBEDDED], 503, id="dropped-connections"),
        # the slowest answer that still gets through: a retry just inside the window, then three 60s calls
        pytest.param([(44, _overloaded()), _ROUTED, _ANSWERED], [_EMBEDDED], 200, id="slow-but-answered"),
    ],
)
def test_a_chat_is_over_before_the_mcp_proxy_gives_up(monkeypatch, clock, generate, embed, status):
    models = SimpleNamespace(generate_content=_gemini_method(clock, generate), embed_content=_gemini_method(clock, embed))
    monkeypatch.setattr(generation, "get_gemini_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(embeddings, "get_gemini_client", lambda: SimpleNamespace(models=models))
    monkeypatch.setattr(retrieval, "search", _no_chunks)
    monkeypatch.setattr(routes, "log_event", lambda **fields: None)

    response = TestClient(app).post("/chat", json={"query": "refund?"})

    # answered or refused, the MCP proxy is still waiting to pass it on
    assert clock.now <= _MCP_PROXY_TIMEOUT_S
    assert response.status_code == status


async def test_earlier_turns_are_sent_before_the_latest_message(monkeypatch):
    sent = {}

    def fake_generate_sync(**kwargs):
        sent.update(kwargs)
        return "response"

    monkeypatch.setattr(generation, "_generate_sync", fake_generate_sync)
    history = [
        types.Content(role="user", parts=[types.Part(text="earlier question")]),
        types.Content(role="model", parts=[types.Part(text="earlier answer")]),
    ]

    await generation.generate(prompt="latest", system_instruction="sys", history=history)

    assert [(c.role, c.parts[0].text) for c in sent["contents"]] == [
        ("user", "earlier question"), ("model", "earlier answer"), ("user", "latest"),
    ]


def test_geminis_own_504_is_a_timeout_and_is_not_retried(client):
    models, sleeps = client([ServerError(504, {"error": {"code": 504, "message": "Deadline expired", "status": "DEADLINE_EXCEEDED"}})])

    with pytest.raises(ModelTimeoutError):
        _call()
    assert models.calls == 1 and sleeps == []
