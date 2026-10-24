"""Visitors' ratings of answers: a thumbs up or down from the chat widget.

A rating saves the rated question and answer, and only when the visitor
chooses to rate. That is what makes it useful: a thumbs-down is a real
question the agent got wrong, a candidate for the eval set
(scripts/feedback_candidates.py). Nothing identifies the visitor.

The text comes from the server's own record of the answer it gave, kept for a
day under its answer_id, not from the request: the widget used to send the
question and answer, so anyone could post made-up pairs that became draft eval
cases. An answer can be rated once."""

import json
import secrets
from collections import deque
from datetime import datetime, timezone

from fastapi import APIRouter, Header, HTTPException, Request, Response
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from app.api.answer_cache import MemoryAnswerCache, RedisAnswerCache, answer_cache_key
from app.api.ratelimit import rate_limit
from app.core.config import Settings, get_settings
from app.core.observability import log_event
from app.core.redis_client import KEY_PREFIX, RedisOutage
from app.models.schemas import FeedbackRequest

MAX_KEPT = 2000  # the newest ratings kept; older ones are dropped
ISSUED_TTL_S = 24 * 3600  # how long an answer can be rated
MAX_ISSUED = 5000  # answers remembered in the process (without Redis)
FEEDBACK_KEY = f"{KEY_PREFIX}feedback"
LEGACY_FEEDBACK_KEY = "feedback"  # the list's name before keys carried the app's prefix (October 2026)

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

    def __init__(self, fallback: MemoryFeedbackStore, redis: Redis, outage: RedisOutage | None = None) -> None:
        self._fallback = fallback
        self._redis = redis
        self._outage = outage or RedisOutage()
        self._moved = False

    def _failed(self, exc: RedisError) -> None:
        if self._outage.failed():  # once per outage, not once per request
            log_event(event="redis_unavailable", used_by="feedback", error=str(exc))

    async def _move_legacy_list(self) -> None:
        """Ratings saved before the prefix are real visitors' and have no TTL, so
        they're renamed rather than left behind. RENAMENX does nothing if the new
        list already exists; Redis errors when there's no old list to move."""
        if self._moved:
            return
        try:
            moved = await self._redis.renamenx(LEGACY_FEEDBACK_KEY, FEEDBACK_KEY)
        except ResponseError:  # no such key: nothing to move
            moved = False
        self._moved = True
        if moved:
            log_event(event="feedback_moved", source=LEGACY_FEEDBACK_KEY, target=FEEDBACK_KEY)

    async def add(self, record: dict) -> None:
        if self._outage.skip():
            await self._fallback.add(record)
            return
        try:
            await self._move_legacy_list()
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.lpush(FEEDBACK_KEY, json.dumps(record))
                pipe.ltrim(FEEDBACK_KEY, 0, MAX_KEPT - 1)
                await pipe.execute()
        except RedisError as exc:
            self._failed(exc)
            await self._fallback.add(record)
            return
        self._outage.worked()

    async def newest(self) -> list[dict]:
        if self._outage.skip():
            return await self._fallback.newest()
        try:
            await self._move_legacy_list()
            raw = await self._redis.lrange(FEEDBACK_KEY, 0, -1)
        except RedisError as exc:
            self._failed(exc)
            return await self._fallback.newest()
        self._outage.worked()
        # plus any kept in the process while Redis was down
        return [json.loads(item) for item in raw] + await self._fallback.newest()


def build_issued_answers(
    settings: Settings, redis: Redis | None = None, outage: RedisOutage | None = None
) -> MemoryAnswerCache | RedisAnswerCache:
    """The answers /chat gave in the last day, by answer_id: what a rating rates."""
    memory = MemoryAnswerCache(ISSUED_TTL_S, max_entries=MAX_ISSUED)
    return RedisAnswerCache(memory, redis, outage, prefix=f"{KEY_PREFIX}issued:") if redis else memory


def build_feedback_store(
    settings: Settings, redis: Redis | None = None, outage: RedisOutage | None = None
) -> MemoryFeedbackStore | RedisFeedbackStore:
    memory = MemoryFeedbackStore()
    return RedisFeedbackStore(memory, redis, outage) if redis else memory


@router.post("/feedback", status_code=204, dependencies=[rate_limit("feedback")])
async def feedback(rating: FeedbackRequest, request: Request) -> Response:
    """Rate an answer from POST /chat, by its answer_id. The question, answer
    and sources are the server's record of that answer; any the request sends
    are ignored."""
    issued = request.app.state.issued_answers
    answered = await issued.get(rating.answer_id)
    if answered is None:
        raise HTTPException(
            status_code=404, detail="That answer isn't one this server gave in the last day, or it's been rated already."
        )
    await issued.delete(rating.answer_id)  # one rating per answer
    record = {"answer_id": rating.answer_id, "rating": rating.rating, **answered,
              "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    await request.app.state.feedback.add(record)
    evicted = rating.rating == "down" and await _evict_cached(request, answered)
    text = {"question": answered["question"], "answer": answered["answer"]} if get_settings().log_chat_text else {}
    log_event(event="feedback", answer_id=rating.answer_id, rating=rating.rating, sources=answered["sources"],
              evicted_cached_answer=evicted, **text)
    return Response(status_code=204)


async def _evict_cached(request: Request, answered: dict) -> bool:
    """Drop a thumbs-down answer from the answer cache, so the next visitor
    asking the same first question gets a fresh one instead of the same bad
    answer for up to an hour. Only if the cache still holds exactly the rated
    answer, not a newer one."""
    cache = request.app.state.answer_cache
    if cache is None:
        return False
    key = answer_cache_key(answered["question"])
    cached = await cache.get(key)
    if not cached or cached.get("answer") != answered["answer"]:
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
