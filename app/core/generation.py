import asyncio
import re
import time

from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.pricing import generation_cost_usd
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    RateLimitedError,
    fits_retry_window,
    get_gemini_client,
    is_daily_quota_error,
    call_gemini,
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
    tool_config: types.ToolConfig | None = None,
) -> types.GenerateContentResponse:
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        tool_config=tool_config,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        response_mime_type="application/json" if response_schema else None,
        response_schema=response_schema,
    )
    settings = get_settings()

    for attempt in range(1, _MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return call_gemini(
                lambda: get_gemini_client().models.generate_content(
                    model=settings.generation_model,
                    contents=contents,
                    config=config,
                )
            )
        except ClientError as exc:
            if is_daily_quota_error(exc):
                raise DailyQuotaExhaustedError(str(exc)) from exc
            if exc.code != 429:
                raise
            delay = _rate_limit_retry_delay(exc, fallback=2**attempt)
            if attempt == _MAX_RATE_LIMIT_RETRIES or not fits_retry_window(delay):
                raise RateLimitedError(str(exc)) from exc
            time.sleep(delay)

    raise AssertionError("unreachable")


async def generate(
    *,
    prompt: str,
    system_instruction: str,
    tools: list[types.Tool] | None = None,
    response_schema: type[BaseModel] | None = None,
    history: list[types.Content] | None = None,
) -> types.GenerateContentResponse:
    """`prompt` is the latest user message; `history` holds the conversation's
    earlier turns, oldest first."""
    return await asyncio.to_thread(
        _generate_sync,
        contents=[*(history or []), types.Content(role="user", parts=[types.Part(text=prompt)])],
        system_instruction=system_instruction,
        tools=tools,
        response_schema=response_schema,
    )


async def generate_contents(
    *,
    contents: list[types.Content],
    system_instruction: str,
    tools: list[types.Tool] | None = None,
    tool_config: types.ToolConfig | None = None,
) -> types.GenerateContentResponse:
    """A call on a whole conversation so far, tool calls and their results
    included, for the tool loop."""
    return await asyncio.to_thread(
        _generate_sync,
        contents=contents,
        system_instruction=system_instruction,
        tools=tools,
        tool_config=tool_config,
    )


def record_usage(usage: dict, response: types.GenerateContentResponse) -> None:
    """Fill a time_step's usage dict from a response: tokens (thinking counted as output) and cost."""
    metadata = response.usage_metadata
    if metadata is None:
        return
    usage["input_tokens"] = metadata.prompt_token_count or 0
    usage["output_tokens"] = (metadata.candidates_token_count or 0) + (
        metadata.thoughts_token_count or 0
    )
    usage["cost_usd"] = generation_cost_usd(
        get_settings().generation_model, usage["input_tokens"], usage["output_tokens"]
    )
