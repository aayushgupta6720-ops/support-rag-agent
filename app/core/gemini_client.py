from functools import lru_cache

from google import genai
from google.genai.errors import ClientError

from app.core.config import get_settings


@lru_cache
def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key)


class DailyQuotaExhaustedError(Exception):
    """Gemini's per-day quota is used up; no retry can succeed until it resets."""


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
