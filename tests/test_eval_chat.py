"""scripts/eval_chat: conversations through /chat, with the agent and the judge faked."""

import asyncio
import json

import app.api.routes as routes
from app.api.ratelimit import build_rate_limiters
from app.core.config import get_settings
from app.eval.judge import JudgeResult
from app.main import app
from scripts import eval_chat


def test_conversations_go_through_chat_with_their_history_and_first_questions_hit_the_cache(monkeypatch, tmp_path):
    settings = get_settings()
    for name in ("chat_limit_per_minute", "chat_limit_per_day", "chat_limit_all_per_day"):
        monkeypatch.setattr(settings, name, 0)
    monkeypatch.setattr(app.state, "rate_limiters", build_rate_limiters(settings))
    asked = []

    async def fake_run_agent(query, history=None):
        asked.append((query, len(history or [])))
        await asyncio.sleep(0.05)  # so a cached answer is clearly quicker
        return {"answer": f"answer to {query!r} after {len(history or [])} turns", "sources": ["billing-refunds"],
                "chunks": [], "router_prompt_version": "r", "answer_prompt_version": "g"}

    judged = []

    async def fake_judge(query, reference, answer, history=None, docs=None):
        judged.append((query, len(history or []), len(docs or [])))
        return JudgeResult(correct=True, reasoning="fine", votes=[True])

    monkeypatch.setattr(routes, "run_agent", fake_run_agent)
    monkeypatch.setattr(eval_chat, "judge_answer", fake_judge)
    monkeypatch.setattr(eval_chat, "settings_differing_from_render", lambda: {})
    monkeypatch.setattr(eval_chat, "SUMMARIES_DIR", tmp_path)

    assert asyncio.run(eval_chat.main()) == 0

    summary = json.loads(next(tmp_path.glob("chat_*.json")).read_text())
    conversations = [c for c in eval_chat.load_golden_set() if c.history]
    assert len(summary["conversations"]) == len(conversations) and summary["history_ok"] and summary["cache_ok"]
    # The 3-turn case: two earlier user messages played live, then the question, which sees all four turns.
    backup = next(r for r in summary["conversations"] if r["id"] == "h3_multi_turn_backup_count")
    assert (backup["turns"], backup["history_turns"]) == (3, 4)
    assert ("And how many of those one-time backup codes do I get?", 4) in asked
    assert ("And how many of those one-time backup codes do I get?", 4, 1) in judged  # the cited article, whole
    # Each cache question went to the agent once, though it was asked twice.
    for case_id in eval_chat.CACHE_CASES:
        query = next(c.query for c in eval_chat.load_golden_set() if c.id == case_id)
        assert sum(q == query for q, _ in asked) == 1
