"""Chat history per session, so a follow-up like "what about annual plans?"
can be understood. A session keeps its last few exchanges (question and
answer, not the retrieved docs) and ends once it has been idle for a while.

Sessions are keyed by visitor as well as session_id. A client can choose its
own id, so without the visitor in the key, anyone who guessed or reused an id
could continue someone else's conversation and ask the model to repeat it."""

import json
import time
from collections import OrderedDict
from typing import Callable

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.agent.history import Turn
from app.core.config import Settings
from app.core.observability import log_event


class MemorySessionStore:
    """Sessions in the process, least recently used dropped first past
    max_sessions, so memory stays bounded."""

    def __init__(
        self,
        max_exchanges: int,
        ttl_s: float,
        max_sessions: int = 1_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_exchanges = max_exchanges
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._clock = clock
        self._sessions: OrderedDict[str, tuple[float, list[Turn]]] = OrderedDict()

    async def history(self, key: str) -> list[Turn]:
        expires_at, turns = self._sessions.get(key, (0.0, []))
        if expires_at <= self._clock():
            self._sessions.pop(key, None)
            return []
        return list(turns)

    async def add_exchange(self, key: str, question: str, answer: str) -> None:
        turns = await self.history(key) + [Turn("user", question), Turn("model", answer)]
        self._sessions.pop(key, None)
        self._sessions[key] = (self._clock() + self.ttl_s, turns[-2 * self.max_exchanges:])
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)


class RedisSessionStore:
    """Sessions as Redis lists of JSON turns that expire when idle. If Redis
    fails, the in-process `fallback` is used instead, so a Redis outage costs
    conversations their earlier turns rather than failing the request."""

    def __init__(self, fallback: MemorySessionStore, redis: Redis) -> None:
        self._fallback = fallback
        self._redis = redis
        self._redis_ok = True

    def _failed(self, exc: RedisError) -> None:
        if self._redis_ok:  # once per outage, not once per request
            log_event(event="redis_unavailable", used_by="sessions", error=str(exc))
        self._redis_ok = False

    async def history(self, key: str) -> list[Turn]:
        try:
            raw = await self._redis.lrange(f"session:{key}", 0, -1)
        except RedisError as exc:
            self._failed(exc)
            return await self._fallback.history(key)
        self._redis_ok = True
        return [Turn(**json.loads(item)) for item in raw]

    async def add_exchange(self, key: str, question: str, answer: str) -> None:
        name = f"session:{key}"
        turns = [Turn("user", question), Turn("model", answer)]
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.rpush(name, *(json.dumps({"role": t.role, "text": t.text}) for t in turns))
                pipe.ltrim(name, -2 * self._fallback.max_exchanges, -1)
                pipe.expire(name, int(self._fallback.ttl_s))
                await pipe.execute()
        except RedisError as exc:
            self._failed(exc)
            await self._fallback.add_exchange(key, question, answer)
            return
        self._redis_ok = True


def build_session_store(settings: Settings, redis: Redis | None = None) -> MemorySessionStore | RedisSessionStore:
    memory = MemorySessionStore(settings.session_max_exchanges, settings.session_ttl_s)
    return RedisSessionStore(memory, redis) if redis else memory
