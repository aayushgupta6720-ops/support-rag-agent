"""The Key Value instance is shared with other apps: every key this app writes
carries its prefix, ratings saved under the old unprefixed name are moved, and
the connection pool is capped."""

import fakeredis
from redis.asyncio import BlockingConnectionPool

from app.api.answer_cache import MemoryAnswerCache, RedisAnswerCache
from app.api.feedback import FEEDBACK_KEY, MemoryFeedbackStore, RedisFeedbackStore, build_issued_answers
from app.api.ratelimit import build_rate_limiters
from app.api.sessions import build_session_store
from app.core import redis_client
from app.core.config import get_settings


async def test_every_key_starts_with_the_app_prefix():
    redis = fakeredis.FakeAsyncRedis()
    settings = get_settings()
    await RedisAnswerCache(MemoryAnswerCache(60), redis).put("k", {"answer": "a"})
    await build_issued_answers(settings, redis).put("id", {"answer": "a"})
    await RedisFeedbackStore(MemoryFeedbackStore(), redis).add({"rating": "up"})
    await build_session_store(settings, redis).add_exchange("v|s", "q", "a")
    await build_rate_limiters(settings, redis)["chat"].hit("visitor")
    keys = sorted(k.decode() for k in await redis.keys("*"))
    assert len(keys) == 5 and all(k.startswith("support:") for k in keys), keys


async def test_ratings_saved_before_the_prefix_are_moved_not_lost():
    redis = fakeredis.FakeAsyncRedis()
    await redis.lpush("feedback", b'{"rating": "down", "n": 1}')
    store = RedisFeedbackStore(MemoryFeedbackStore(), redis)
    await store.add({"rating": "up", "n": 2})
    assert [r["n"] for r in await store.newest()] == [2, 1]
    assert await redis.exists("feedback") == 0 and await redis.llen(FEEDBACK_KEY) == 2


async def test_moving_is_skipped_when_there_is_nothing_to_move():
    redis = fakeredis.FakeAsyncRedis()
    store = RedisFeedbackStore(MemoryFeedbackStore(), redis)
    assert await store.newest() == []
    await store.add({"rating": "up"})
    assert await redis.llen(FEEDBACK_KEY) == 1


def test_the_client_caps_its_connections(monkeypatch):
    monkeypatch.setattr(get_settings(), "redis_url", "redis://localhost:6379/0")
    redis_client.get_redis.cache_clear()
    try:
        pool = redis_client.get_redis().connection_pool
        assert isinstance(pool, BlockingConnectionPool) and pool.max_connections == 10
    finally:
        redis_client.get_redis.cache_clear()
