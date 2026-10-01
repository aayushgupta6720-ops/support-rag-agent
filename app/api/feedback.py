"""Visitors' ratings of answers: a thumbs up or down from the chat widget.

A rating saves the rated question and answer, which the visitor's own browser
sends and only when they choose to rate. That is what makes it useful: a
thumbs-down is a real question the agent got wrong, a candidate for the eval
set (scripts/feedback_candidates.py). Nothing identifies the visitor."""

import json
import secrets
from collections import deque
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request, Response
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.api.answer_cache import answer_cache_key
from app.api.ratelimit import rate_limit
from app.core.config import Settings, get_settings
from app.core.observability import log_event
from app.models.schemas import FeedbackRequest

MAX_KEPT = 2000  # the newest ratings kept; older ones are dropped

router = APIRouter()


class MemoryFeedbackStore:
    def __init__(self, max_kept: int = MAX_KEPT) -> None:
        self._records: deque[dict] = deque(maxlen=max_kept)

    async def add(self, record: dict) -> None:
        self._records.appendleft(record)

    async def newest(self) -> list[dict]:
        return list(self._records)


class RedisFeedbackStore:
    """Ratings as a capped Redis list, newest first. If Redis fails, the
    in-process `fallback` is used instead."""

    def __init__(self, fallback: MemoryFeedbackStore, redis: Redis) -> None:
        self._fallback = fallback
        self._redis = redis
        self._redis_ok = True

    def _failed(self, exc: RedisError) -> None:
        if self._redis_ok:  # once per outage, not once per request
            log_event(event="redis_unavailable", used_by="feedback", error=str(exc))
        self._redis_ok = False

    async def add(self, record: dict) -> None:
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.lpush("feedback", json.dumps(record))
                pipe.ltrim("feedback", 0, MAX_KEPT - 1)
                await pipe.execute()
        except RedisError as exc:
            self._failed(exc)
            await self._fallback.add(record)
            return
        self._redis_ok = True

    async def newest(self) -> list[dict]:
        try:
            raw = await self._redis.lrange("feedback", 0, -1)
        except RedisError as exc:
            self._failed(exc)
            return await self._fallback.newest()
        self._redis_ok = True
        # plus any kept in the process while Redis was down
        return [json.loads(item) for item in raw] + await self._fallback.newest()


def build_feedback_store(settings: Settings, redis: Redis | None = None) -> MemoryFeedbackStore | RedisFeedbackStore:
    memory = MemoryFeedbackStore()
    return RedisFeedbackStore(memory, redis) if redis else memory


@router.post("/feedback", status_code=204, dependencies=[rate_limit("feedback")])
async def feedback(rating: FeedbackRequest, request: Request) -> Response:
    """Rate an answer from POST /chat, by its answer_id."""
    record = rating.model_dump() | {"received_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    await request.app.state.feedback.add(record)
    evicted = rating.rating == "down" and await _evict_cached(request, rating)
    text = {"question": rating.question, "answer": rating.answer} if get_settings().log_chat_text else {}
    log_event(event="feedback", answer_id=rating.answer_id, rating=rating.rating, sources=rating.sources,
              evicted_cached_answer=evicted, **text)
    return Response(status_code=204)


async def _evict_cached(request: Request, rating: FeedbackRequest) -> bool:
    """Drop a thumbs-down answer from the answer cache, so the next visitor
    asking the same first question gets a fresh one instead of the same bad
    answer for up to an hour. Only if the cache still holds exactly the rated
    answer: a made-up rating can't evict a different one."""
    cache = request.app.state.answer_cache
    if cache is None:
        return False
    key = answer_cache_key(rating.question)
    cached = await cache.get(key)
    if not cached or cached.get("answer") != rating.answer:
        return False
    await cache.delete(key)
    return True


@router.get("/feedback/export", include_in_schema=False)
async def export_feedback(request: Request, authorization: str = Header(default="")) -> dict:
    """Every kept rating, newest first, for scripts/feedback_candidates.py.
    Needs FEEDBACK_EXPORT_TOKEN as a bearer token; without one configured,
    the endpoint doesn't exist."""
    token = get_settings().feedback_export_token
    if not token:
        raise HTTPException(status_code=404, detail="Not Found")
    if not secrets.compare_digest(authorization.encode(), f"Bearer {token}".encode()):
        raise HTTPException(status_code=401, detail="A valid export token is required.",
                            headers={"WWW-Authenticate": "Bearer"})
    return {"feedback": await request.app.state.feedback.newest()}
