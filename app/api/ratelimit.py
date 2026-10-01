"""Per-visitor request limits at the HTTP edge, so one visitor can't use up
the shared daily Gemini quota for everyone. A visitor is an IP address (for
IPv6, its /64, since one visitor usually holds all 2^64 of those).

With REDIS_URL set, counts live in Redis, so they survive restarts (on
Render's free plan, the service restarts every time it wakes from sleep) and
would be shared by several instances. Without it, or while Redis is
unreachable, they live in the process."""

import ipaddress
import math
import secrets
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, HTTPException, Request
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import Settings, get_settings
from app.core.observability import log_event


@dataclass(frozen=True)
class Limit:
    count: int
    window_s: float
    per: str  # how the window reads in a message: "a minute", "a day"


class RateLimiter:
    """Sliding-window limits per key. A key keeps the times of its recent
    requests, never more than its largest limit allows, and at most
    max_keys keys are tracked (least recently seen dropped first), so memory
    stays bounded."""

    def __init__(
        self,
        what: str,
        limits: list[Limit],
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.what = what  # what's being counted, for the refusal: "messages"
        self.limits = [limit for limit in limits if limit.count > 0]  # 0 turns one off
        self.max_keys = max_keys
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    async def hit(self, key: str) -> tuple[Limit, float] | None:
        """Count a request from `key`. If a limit is already reached, count
        nothing and return that limit and the seconds until it allows another
        request: the longest wait, when more than one is reached. Async only
        to match RedisRateLimiter: it never awaits, so concurrent requests
        can't interleave inside it."""
        if not self.limits:
            return None
        now = self._clock()
        hits = self._hits.pop(key, deque())
        longest = max(limit.window_s for limit in self.limits)
        while hits and now - hits[0] >= longest:
            hits.popleft()
        self._hits[key] = hits  # re-inserted last: the most recently seen key

        refusals = []
        for limit in self.limits:
            in_window = [t for t in hits if now - t < limit.window_s]
            if len(in_window) >= limit.count:
                refusals.append((limit, in_window[-limit.count] + limit.window_s - now))
        if refusals:
            return max(refusals, key=lambda refusal: refusal[1])

        hits.append(now)
        while len(self._hits) > self.max_keys:
            self._hits.popitem(last=False)
        return None


# The same sliding window as RateLimiter.hit, run inside Redis so the check
# and the count are one atomic step. KEYS[1] is the visitor's sorted set of
# request times; ARGV is now, a unique member for this request, then count
# and window for each limit. Returns {index of the refusing limit (1-based,
# 0 if none), wait in seconds}. The wait is a string because Redis truncates
# Lua numbers to integers.
_SLIDING_WINDOW = """
local now = tonumber(ARGV[1])
local longest = 0
for i = 3, #ARGV, 2 do
  longest = math.max(longest, tonumber(ARGV[i + 1]))
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - longest)
local refused, wait = 0, 0
for i = 3, #ARGV, 2 do
  local count, window = tonumber(ARGV[i]), tonumber(ARGV[i + 1])
  if redis.call('ZCOUNT', KEYS[1], '(' .. (now - window), '+inf') >= count then
    local nth = redis.call('ZREVRANGE', KEYS[1], count - 1, count - 1, 'WITHSCORES')
    local this_wait = tonumber(nth[2]) + window - now
    if refused == 0 or this_wait > wait then
      refused, wait = (i - 1) / 2, this_wait
    end
  end
end
if refused > 0 then
  return {refused, tostring(wait)}
end
redis.call('ZADD', KEYS[1], now, ARGV[2])
redis.call('EXPIRE', KEYS[1], math.ceil(longest))
return {0, '0'}
"""


class RedisRateLimiter:
    """RateLimiter's limits, counted in Redis. If Redis fails, requests are
    counted by the in-process `fallback` instead: still limited, just not
    shared, rather than either refusing everyone or letting everyone through."""

    def __init__(self, fallback: RateLimiter, redis: Redis, clock: Callable[[], float] = time.time) -> None:
        self.what = fallback.what
        self.limits = fallback.limits
        self._fallback = fallback
        self._script = redis.register_script(_SLIDING_WINDOW)
        # Wall-clock time, unlike RateLimiter: the counts outlive the process.
        self._clock = clock
        self._redis_ok = True

    async def hit(self, key: str) -> tuple[Limit, float] | None:
        if not self.limits:
            return None
        now = self._clock()
        args = [now, f"{now}:{secrets.token_hex(4)}"]
        for limit in self.limits:
            args += [limit.count, limit.window_s]
        try:
            refused, wait = await self._script(keys=[f"ratelimit:{self.what}:{key}"], args=args)
        except RedisError as exc:
            if self._redis_ok:  # once per outage, not once per request
                log_event(event="redis_unavailable", used_by="rate_limit", error=str(exc))
            self._redis_ok = False
            return await self._fallback.hit(key)
        self._redis_ok = True
        if int(refused) == 0:
            return None
        return self.limits[int(refused) - 1], float(wait)


def build_rate_limiters(settings: Settings, redis: Redis | None = None) -> dict[str, RateLimiter | RedisRateLimiter]:
    """One limiter per costly endpoint. A /chat call is up to three model
    calls: route, embed, generate. Each limiter counts under its own name
    (`what`) in Redis."""
    chat = RateLimiter("questions", [
        Limit(settings.chat_limit_per_minute, 60, "a minute"),
        Limit(settings.chat_limit_per_day, 24 * 3600, "a day"),
    ])
    # Ratings cost no model calls, but each one is stored: enough for a
    # visitor to rate every answer they could get, not enough to flood it.
    feedback = RateLimiter("ratings", [Limit(20, 60, "a minute"), Limit(100, 24 * 3600, "a day")])
    # /health/search makes a Qdrant request but no model call; a scheduler
    # needs one now and then, not a stream.
    search_check = RateLimiter("search checks", [Limit(6, 60, "a minute")])
    limiters = {"chat": chat, "feedback": feedback, "search_check": search_check}
    return {name: RedisRateLimiter(limiter, redis) if redis else limiter for name, limiter in limiters.items()}


_warned_missing_header = False


def client_key(request: Request) -> str:
    """Who's asking. The connecting address, unless CLIENT_IP_HEADER names a
    header that a trusted proxy in front of the app sets (and overwrites if the
    client sent one). Never X-Forwarded-For: proxies append to it, so its first
    entry is whatever the visitor chose."""
    global _warned_missing_header
    header = get_settings().client_ip_header
    ip = request.headers.get(header, "").strip() if header else ""
    if header and not ip:
        # Everyone without the header shares one count: strict, but safe.
        # Falling back to the connecting address isn't, since uvicorn may
        # already have replaced it with X-Forwarded-For's first entry (Render
        # sets FORWARDED_ALLOW_IPS=*, which tells it to).
        if not _warned_missing_header:
            _warned_missing_header = True
            log_event(event="client_ip_header_missing", header=header)
        return f"missing {header}"
    ip = ip or (request.client.host if request.client else "unknown")
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


def _duration(seconds: int) -> str:
    if seconds < 90:
        n, unit = seconds, "second"
    elif seconds < 90 * 60:
        n, unit = math.ceil(seconds / 60), "minute"
    else:
        n, unit = math.ceil(seconds / 3600), "hour"
    return f"{n} {unit}{'' if n == 1 else 's'}"


def rate_limit(name: str):
    """A route dependency that refuses the request with a 429 once this
    visitor is over the `name` limits. Async, so it runs on the event loop:
    FastAPI runs a sync dependency in a threadpool, where requests arriving
    together interleaved inside RateLimiter.hit and got past the limit."""

    async def check(request: Request) -> None:
        limiter: RateLimiter | RedisRateLimiter = request.app.state.rate_limiters[name]
        key = client_key(request)
        refused = await limiter.hit(key)
        if refused is None:
            return
        limit, retry_after_s = refused
        wait = max(1, math.ceil(retry_after_s))
        log_event(event="rate_limited", limiter=name, client=key, retry_after_s=wait)
        raise HTTPException(
            status_code=429,
            detail=f"You've reached the limit of {limit.count} {limiter.what} {limit.per}. "
            f"Try again in {_duration(wait)}.",
            headers={"Retry-After": str(wait)},
        )

    return Depends(check)
