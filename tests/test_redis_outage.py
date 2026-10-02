"""A Redis outage: after one Redis error, every store uses its in-process
fallback for a while instead of each call waiting out the 2s timeout."""

import fakeredis

import app.api.answer_cache as answer_cache
import app.api.feedback as feedback
import app.api.ratelimit as ratelimit
import app.api.sessions as sessions
from app.agent.history import Turn
from app.api.answer_cache import MemoryAnswerCache, RedisAnswerCache
from app.api.feedback import MemoryFeedbackStore, RedisFeedbackStore
from app.api.ratelimit import Limit, RateLimiter, RedisRateLimiter
from app.api.sessions import MemorySessionStore, RedisSessionStore
from app.core.redis_client import RedisOutage


async def test_after_a_redis_error_every_store_skips_redis_for_30s(monkeypatch):
    events = []
    for module in (answer_cache, feedback, ratelimit, sessions):
        monkeypatch.setattr(module, "log_event", lambda **fields: events.append(fields))
    now = [0.0]
    outage = RedisOutage(pause_s=30, clock=lambda: now[0])
    server = fakeredis.FakeServer()
    redis = fakeredis.FakeAsyncRedis(server=server)
    history = RedisSessionStore(MemorySessionStore(3, 1800), redis, outage)
    cache = RedisAnswerCache(MemoryAnswerCache(3600), redis, outage)
    ratings = RedisFeedbackStore(MemoryFeedbackStore(), redis, outage)
    limiter = RedisRateLimiter(RateLimiter("questions", [Limit(6, 60, "a minute")]), redis, outage=outage)

    server.connected = False
    await history.add_exchange("v|s", "q", "a")  # fails, so it's kept in the process
    server.connected = True  # Redis is back, but nothing tries it yet

    now[0] = 29
    await cache.put("k", {"answer": "a"})
    await ratings.add({"answer_id": "x"})
    assert await limiter.hit("v") is None
    assert await history.history("v|s") == [Turn("user", "q"), Turn("model", "a")]
    assert await redis.keys("*") == []  # every store used its fallback

    now[0] = 30
    await history.add_exchange("v|s", "q2", "a2")
    assert await redis.llen("session:v|s") == 2  # 30s on, Redis is tried again
    assert [e["event"] for e in events] == ["redis_unavailable"]  # once for all the stores


def test_an_outage_is_logged_once_and_the_next_one_again():
    now = [0.0]
    outage = RedisOutage(pause_s=30, clock=lambda: now[0])
    assert not outage.skip()

    assert outage.failed()  # the first error: log it
    assert outage.skip()
    now[0] = 30
    assert not outage.skip()  # time to try Redis again
    assert not outage.failed()  # still down: already logged, skip another 30s
    assert outage.skip()

    outage.worked()
    now[0] = 100
    assert outage.failed()  # a new outage
