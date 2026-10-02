import time
from functools import lru_cache
from typing import Callable

from redis.asyncio import Redis

from app.core.config import get_settings


@lru_cache
def get_redis() -> Redis | None:
    """The shared Redis client, or None when REDIS_URL is unset. Timeouts are
    short because every caller falls back to in-process state when Redis
    fails, and RedisOutage then skips Redis for a while: a slow Redis should
    cost one request a couple of seconds, not hang it or every one after it."""
    url = get_settings().redis_url
    if not url:
        return None
    return Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2, health_check_interval=30)


class RedisOutage:
    """Whether Redis is down, shared by every store that uses it. After a
    Redis error, the stores go straight to their in-process fallbacks for
    the next `pause_s` seconds, then try Redis again. Without this, each
    call during an outage waited out the 2s timeout first, and a /chat
    makes up to five of them: every /chat took 10s longer while Redis hung."""

    def __init__(self, pause_s: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.pause_s = pause_s
        self._clock = clock
        self._skip_until = 0.0
        self._ok = True

    def skip(self) -> bool:
        """True within pause_s of a Redis error: use the fallback without trying Redis."""
        return self._clock() < self._skip_until

    def failed(self) -> bool:
        """Records a Redis error. True if Redis had been working until now, so
        an outage is logged once, not once per call."""
        first = self._ok
        self._ok = False
        self._skip_until = self._clock() + self.pause_s
        return first

    def worked(self) -> None:
        self._ok = True
