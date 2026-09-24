from types import SimpleNamespace

import pytest
from google.genai.errors import ClientError, ServerError

import app.core.generation as generation
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    RateLimitedError,
    is_daily_quota_error,
    next_daily_quota_reset,
)
from tests.fakes import DAILY, PER_MINUTE, rate_limited_error


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
        ServerError(500, {"error": {"code": 500, "message": "boom"}}),
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
