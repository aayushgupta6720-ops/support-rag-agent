from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError, ServerError

import app.core.generation as generation


def _rate_limited(retry_delay: str | None = "7s") -> ClientError:
    details = []
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return ClientError(429, {"error": {"code": 429, "message": "quota", "details": details}})


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
    assert generation._rate_limit_retry_delay(_rate_limited("7s"), fallback=1) == 7.0
    assert generation._rate_limit_retry_delay(_rate_limited("0.5s"), fallback=1) == 0.5


def test_retry_delay_falls_back_without_retry_info():
    assert generation._rate_limit_retry_delay(_rate_limited(None), fallback=4) == 4


def test_retries_rate_limit_then_succeeds(client):
    models, sleeps = client([_rate_limited("3s"), _rate_limited(None), "response"])

    assert _call() == "response"
    assert models.calls == 3
    # server-suggested delay first, then exponential fallback (2**attempt)
    assert sleeps == [3.0, 4]


def test_gives_up_after_max_retries(client):
    models, sleeps = client([_rate_limited()] * generation._MAX_RATE_LIMIT_RETRIES)

    with pytest.raises(ClientError):
        _call()
    assert models.calls == generation._MAX_RATE_LIMIT_RETRIES
    assert len(sleeps) == generation._MAX_RATE_LIMIT_RETRIES - 1


@pytest.mark.parametrize(
    "error",
    [
        ClientError(400, {"error": {"code": 400, "message": "bad request"}}),
        ServerError(500, {"error": {"code": 500, "message": "boom"}}),
    ],
)
def test_other_errors_are_not_retried(client, error):
    models, sleeps = client([error, "never reached"])

    with pytest.raises(type(error)):
        _call()
    assert models.calls == 1
    assert sleeps == []
