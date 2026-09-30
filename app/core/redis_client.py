from functools import lru_cache

from redis.asyncio import Redis

from app.core.config import get_settings


@lru_cache
def get_redis() -> Redis | None:
    """The shared Redis client, or None when REDIS_URL is unset. Timeouts are
    short because every caller falls back to in-process state when Redis
    fails: a slow Redis should cost a request a couple of seconds, not hang it."""
    url = get_settings().redis_url
    if not url:
        return None
    return Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2, health_check_interval=30)
