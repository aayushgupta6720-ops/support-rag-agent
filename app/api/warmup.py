"""Answers to the help center's suggested questions, made before anyone asks.

The topic cards and chips (app/static/index.html and chat.js) send the same
fifteen or so questions, usually as a visitor's first, and each costs two
Gemini calls and a few seconds. A minute after startup, any of them missing
from the answer cache is answered once, a few seconds apart, and counted
against the daily budget like a visitor's question. Cached answers last a week
in Redis and their key changes with the articles, prompts and models
(app/api/answer_cache.py), so once they're in this costs nothing until one of
those changes.

Without Redis, or while it's down, it does nothing: an in-process cache is
lost at every restart, and refilling it each time the free instance wakes
would spend the quota."""

import asyncio
import html
import re
from functools import lru_cache
from pathlib import Path

from app.agent.graph import run_agent
from app.api.answer_cache import RedisAnswerCache, answer_cache_key, cached_answer
from app.api.ratelimit import EVERYONE
from app.api.routes import RETRY_WINDOW_S
from app.core.config import get_settings
from app.core.gemini_client import start_retry_window
from app.core.observability import log_event, start_trace

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@lru_cache
def suggested_questions() -> tuple[str, ...]:
    """The questions the page's cards and the widget's chips send, in page order."""
    page = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    script = (STATIC_DIR / "chat.js").read_text(encoding="utf-8")
    asks = [html.unescape(question) for question in re.findall(r'data-ask="([^"]+)"', page)]
    block = re.search(r"const SUGGESTIONS = \[(.*?)\];", script, re.S)
    chips = re.findall(r'"([^"\\]+)"', block.group(1)) if block else []
    return tuple(dict.fromkeys(asks + chips))


async def warm_suggested_answers(state, delay_s: float = 60.0, gap_s: float = 5.0) -> int:
    """Answer the suggested questions that aren't cached; returns how many were.
    `state` is the app's state: its answer cache and rate limiters."""
    cache = state.answer_cache
    if not isinstance(cache, RedisAnswerCache) or not get_settings().gemini_api_key:
        return 0
    await asyncio.sleep(delay_s)  # the request that woke the instance goes first
    warmed = 0
    for question in suggested_questions():
        key = answer_cache_key(question)
        cached = await cache.get(key)
        if cache.redis_down():
            log_event(event="answer_warmup", stopped="redis_unavailable", warmed=warmed)
            return warmed
        if cached is not None:
            continue
        if await state.rate_limiters["chat_all"].hit(EVERYONE) is not None:
            log_event(event="answer_warmup", stopped="daily_budget", warmed=warmed)
            return warmed
        start_trace()
        start_retry_window(RETRY_WINDOW_S)
        try:
            result = await run_agent(question)
        except Exception as exc:  # noqa: BLE001 - the quota or Gemini failing ends the warm-up, never the app
            log_event(event="answer_warmup", stopped=type(exc).__name__, error=str(exc), warmed=warmed)
            return warmed
        await cache.put(key, cached_answer(result))
        warmed += 1
        await asyncio.sleep(gap_s)  # stay well under the per-minute quota
    log_event(event="answer_warmup", warmed=warmed, already_cached=len(suggested_questions()) - warmed)
    return warmed
