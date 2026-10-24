"""Answering the suggested questions before anyone asks (app/api/warmup.py)."""

from types import SimpleNamespace

import fakeredis
import pytest

import app.api.warmup as warmup
from app.api.answer_cache import MemoryAnswerCache, RedisAnswerCache, answer_cache_key
from app.api.ratelimit import Limit, RateLimiter
from app.core.config import get_settings
from app.core.gemini_client import DailyQuotaExhaustedError


def test_the_questions_come_from_the_page_and_the_widget():
    questions = warmup.suggested_questions()
    assert "How do I reset my password?" in questions  # a topic card (index.html)
    assert "What are the API rate limits?" in questions  # a chip (chat.js)
    assert len(questions) == len(set(questions)) >= 10


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setattr(get_settings(), "gemini_api_key", "test-key")
    asked: list[str] = []

    async def run_agent(question, history=None):
        asked.append(question)
        return {"answer": f"answer to {question}", "sources": ["password-reset"], "chunks": []}

    monkeypatch.setattr(warmup, "run_agent", run_agent)
    monkeypatch.setattr(warmup, "suggested_questions", lambda: ("First?", "Second?", "Third?"))
    return asked


def _state(cache, budget: int = 100):
    return SimpleNamespace(answer_cache=cache,
                           rate_limiters={"chat_all": RateLimiter("questions from everyone", [Limit(budget, 86400, "a day")])})


async def test_only_the_missing_answers_are_made_and_cached(agent):
    cache = RedisAnswerCache(MemoryAnswerCache(60), fakeredis.FakeAsyncRedis())
    await cache.put(answer_cache_key("Second?"), {"answer": "already here"})
    assert await warmup.warm_suggested_answers(_state(cache), delay_s=0, gap_s=0) == 2
    assert agent == ["First?", "Third?"]
    assert (await cache.get(answer_cache_key("Third?")))["answer"] == "answer to Third?"
    assert await warmup.warm_suggested_answers(_state(cache), delay_s=0, gap_s=0) == 0  # the next start costs nothing


async def test_it_stops_at_the_daily_budget(agent):
    cache = RedisAnswerCache(MemoryAnswerCache(60), fakeredis.FakeAsyncRedis())
    assert await warmup.warm_suggested_answers(_state(cache, budget=1), delay_s=0, gap_s=0) == 1
    assert agent == ["First?"]


async def test_it_stops_when_the_model_fails(agent, monkeypatch):
    async def quota_used(question, history=None):
        raise DailyQuotaExhaustedError("daily quota")

    monkeypatch.setattr(warmup, "run_agent", quota_used)
    cache = RedisAnswerCache(MemoryAnswerCache(60), fakeredis.FakeAsyncRedis())
    assert await warmup.warm_suggested_answers(_state(cache), delay_s=0, gap_s=0) == 0


async def test_it_does_nothing_without_redis_or_while_redis_is_down(agent):
    assert await warmup.warm_suggested_answers(_state(MemoryAnswerCache(60)), delay_s=0, gap_s=0) == 0
    server = fakeredis.FakeServer()
    server.connected = False
    down = RedisAnswerCache(MemoryAnswerCache(60), fakeredis.FakeAsyncRedis(server=server))
    assert await warmup.warm_suggested_answers(_state(down), delay_s=0, gap_s=0) == 0
    assert agent == []
