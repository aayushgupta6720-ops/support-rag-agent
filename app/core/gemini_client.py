import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import TypeVar
from zoneinfo import ZoneInfo

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from app.core.config import get_settings

_PACIFIC = ZoneInfo("America/Los_Angeles")
# A 503 "high demand" usually clears within seconds to minutes: retry a few
# times (1s, 2s, 4s) rather than make a user wait out a long spike.
_MAX_OVERLOAD_ATTEMPTS = 4

T = TypeVar("T")


@lru_cache
def get_gemini_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(
        api_key=settings.gemini_api_key,
        # Milliseconds, applied to connecting and to each read, so a stream
        # that stalls partway is caught as well as one that never starts.
        http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_s * 1000)),
    )


class DailyQuotaExhaustedError(Exception):
    """Gemini's per-day quota is used up; no retry can succeed until it resets."""


class ModelOverloadedError(Exception):
    """Gemini kept answering 503 UNAVAILABLE ("high demand") past our
    retries: a Google-side capacity spike, not anything wrong with the call."""


class ModelTimeoutError(Exception):
    """A Gemini call got no response within gemini_timeout_s: usually a
    Google-side slowdown rather than anything wrong with the call."""


def call_gemini(call: Callable[[], T]) -> T:
    """call(), with Gemini's transient failures handled:
    - 503 UNAVAILABLE, which Google says is usually temporary, is retried
      with a short backoff and becomes ModelOverloadedError once the retries
      run out.
    - A call that times out becomes ModelTimeoutError straight away, since a
      retry would double an already long wait.
    Other errors pass through."""
    for attempt in range(1, _MAX_OVERLOAD_ATTEMPTS + 1):
        try:
            return call()
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(
                f"no response from Gemini within {get_settings().gemini_timeout_s:g}s"
            ) from exc
        except ServerError as exc:
            if exc.code != 503:
                raise
            if attempt == _MAX_OVERLOAD_ATTEMPTS:
                raise ModelOverloadedError(str(exc)) from exc
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


class RateLimitedError(Exception):
    """Gemini kept rate-limiting a call (a per-minute quota) past our retries.
    Unlike DailyQuotaExhaustedError, trying again in a minute can work."""


def next_daily_quota_reset(now: datetime | None = None) -> datetime:
    """When Gemini's per-day quotas next reset: midnight Pacific time."""
    pacific_now = (now or datetime.now(timezone.utc)).astimezone(_PACIFIC)
    tomorrow = pacific_now.date() + timedelta(days=1)
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=_PACIFIC)


def is_daily_quota_error(exc: ClientError) -> bool:
    """True for a 429 caused by a per-day quota (e.g. the free tier's
    requests-per-day cap). Its RetryInfo still suggests ~60s, so a retry loop
    that trusts it just sleeps and fails again until the quota resets at
    midnight Pacific; per-minute 429s are worth retrying, these aren't."""
    if exc.code != 429:
        return False
    details = (exc.details or {}).get("error", {}).get("details", [])
    return any(
        "PerDay" in violation.get("quotaId", "")
        for detail in details
        if detail.get("@type", "").endswith("QuotaFailure")
        for violation in detail.get("violations", [])
    )
