"""Answers to repeat questions. The help center's topic cards and chips send
the same few questions over and over; each would otherwise cost two Gemini
calls and a few seconds.

Only a conversation's first question is cached, since a follow-up depends on
its history. The key is a hash of the normalized question plus everything
that shapes its answer (models, prompt versions, retrieval settings, the
collection), so changing any of them misses instead of serving an answer made
the old way. The question itself is never stored. After re-ingesting changed
docs, cached answers can lag until their TTL runs out."""

import hashlib
import json
import re
import time
from collections import OrderedDict
from typing import Callable

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.agent.prompts import DIRECT_ANSWER_PROMPT_VERSION, GROUNDED_ANSWER_PROMPT_VERSION, ROUTER_PROMPT_VERSION
from app.core.config import Settings, get_settings
from app.core.observability import log_event
from app.core.redis_client import RedisOutage


def answer_cache_key(query: str) -> str:
    settings = get_settings()
    # "How do refunds work?" and "how do refunds work" are the same question
    normalized = re.sub(r"\s+", " ", query.casefold()).strip().rstrip("?!. ")
    shaping = [
        normalized,
        ROUTER_PROMPT_VERSION,
        DIRECT_ANSWER_PROMPT_VERSION,
        GROUNDED_ANSWER_PROMPT_VERSION,
        settings.generation_model,
        settings.embedding_model,
        settings.qdrant_collection,
        str(settings.retrieval_top_k),
        str(settings.retrieval_min_score),
        str(settings.retrieval_max_score_gap),
    ]
    return hashlib.sha256("\x1f".join(shaping).encode()).hexdigest()


class MemoryAnswerCache:
    """Answers in the process, least recently used dropped first past
    max_entries, so memory stays bounded."""

    def __init__(self, ttl_s: float, max_entries: int = 500, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, dict]] = OrderedDict()

    async def get(self, key: str) -> dict | None:
        expires_at, value = self._entries.get(key, (0.0, None))
        if expires_at <= self._clock():
            self._entries.pop(key, None)
            return None
        self._entries.move_to_end(key)
        return value

    async def put(self, key: str, value: dict) -> None:
        self._entries.pop(key, None)
        self._entries[key] = (self._clock() + self.ttl_s, value)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)


class RedisAnswerCache:
    """Answers in Redis, expiring after the TTL. If Redis fails, the
    in-process `fallback` is used instead."""

    def __init__(self, fallback: MemoryAnswerCache, redis: Redis, outage: RedisOutage | None = None) -> None:
        self._fallback = fallback
        self._redis = redis
        self._outage = outage or RedisOutage()

    def _failed(self, exc: RedisError) -> None:
        if self._outage.failed():  # once per outage, not once per request
            log_event(event="redis_unavailable", used_by="answer_cache", error=str(exc))

    async def get(self, key: str) -> dict | None:
        if self._outage.skip():
            return await self._fallback.get(key)
        try:
            raw = await self._redis.get(f"answer:{key}")
        except RedisError as exc:
            self._failed(exc)
            return await self._fallback.get(key)
        self._outage.worked()
        return json.loads(raw) if raw else None

    async def put(self, key: str, value: dict) -> None:
        if self._outage.skip():
            await self._fallback.put(key, value)
            return
        try:
            await self._redis.set(f"answer:{key}", json.dumps(value), ex=int(self._fallback.ttl_s))
        except RedisError as exc:
            self._failed(exc)
            await self._fallback.put(key, value)
            return
        self._outage.worked()

    async def delete(self, key: str) -> None:
        await self._fallback.delete(key)  # it may hold a copy from a Redis outage
        if self._outage.skip():
            return
        try:
            await self._redis.delete(f"answer:{key}")
        except RedisError as exc:
            self._failed(exc)
            return
        self._outage.worked()


def build_answer_cache(
    settings: Settings, redis: Redis | None = None, outage: RedisOutage | None = None
) -> MemoryAnswerCache | RedisAnswerCache | None:
    """None when ANSWER_CACHE_TTL_S is 0 (caching off)."""
    if settings.answer_cache_ttl_s <= 0:
        return None
    memory = MemoryAnswerCache(settings.answer_cache_ttl_s)
    return RedisAnswerCache(memory, redis, outage) if redis else memory
