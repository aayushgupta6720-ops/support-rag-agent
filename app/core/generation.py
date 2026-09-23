import asyncio
import re
import time

from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    get_gemini_client,
    is_daily_quota_error,
)

_MAX_RATE_LIMIT_RETRIES = 5


def _rate_limit_retry_delay(exc: ClientError, fallback: float) -> float:
    details = (exc.details or {}).get("error", {}).get("details", [])
    for detail in details:
        if detail.get("@type", "").endswith("RetryInfo"):
            match = re.match(r"([\d.]+)s?", detail.get("retryDelay", ""))
            if match:
                return float(match.group(1))
    return fallback


def _generate_sync(
    *,
    contents: list[types.Content],
    system_instruction: str,
    tools: list[types.Tool] | None = None,
    response_schema: type[BaseModel] | None = None,
) -> types.GenerateContentResponse:
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        response_mime_type="application/json" if response_schema else None,
        response_schema=response_schema,
    )
    settings = get_settings()

    for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return get_gemini_client().models.generate_content(
                model=settings.generation_model,
                contents=contents,
                config=config,
            )
        except ClientError as exc:
            if is_daily_quota_error(exc):
                raise DailyQuotaExhaustedError(str(exc)) from exc
            if exc.code != 429 or attempt == _MAX_RATE_LIMIT_RETRIES:
                raise
            time.sleep(_rate_limit_retry_delay(exc, fallback=2**attempt))

    raise AssertionError("unreachable")


async def generate(
    *,
    prompt: str,
    system_instruction: str,
    tools: list[types.Tool] | None = None,
    response_schema: type[BaseModel] | None = None,
) -> types.GenerateContentResponse:
    return await asyncio.to_thread(
        _generate_sync,
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        system_instruction=system_instruction,
        tools=tools,
        response_schema=response_schema,
    )
