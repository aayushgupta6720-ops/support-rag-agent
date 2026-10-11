"""The golden set's conversations through the real /chat endpoint.

    python -m scripts.eval_chat

scripts/eval.py calls the agent directly, with scripted earlier turns. This goes
through the API the widget uses, in process: each conversation case is played
live, its earlier user messages sent one by one in one session so the agent
writes its own earlier answers, then the final question, which the eval's judge
grades against the case's reference. It also checks what only the API does:
that the session hands the agent every earlier turn (history_turns), and that a
conversation's first question, asked again in a new session, comes from the
answer cache unchanged.

Rate limits and the daily budget are switched off and REDIS_URL is emptied
before the app loads, so it uses in-process stores and never the live Redis. A
summary goes to data/eval/summaries/chat_<time>.json. About 75 chat-model
requests.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(CHAT_LIMIT_PER_MINUTE="0", CHAT_LIMIT_PER_DAY="0", CHAT_LIMIT_ALL_PER_DAY="0", REDIS_URL="")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.agent.history import Turn  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.judge import judge_answer  # noqa: E402
from app.main import app  # noqa: E402
from app.rag.documents import load_documents  # noqa: E402
from scripts.eval import _git_commit, settings_differing_from_render  # noqa: E402

SUMMARIES_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "summaries"
CACHE_CASES = ("password_reset_1", "billing_refund_1", "rate_limit_1")


async def ask(client: httpx.AsyncClient, query: str, session_id: str | None) -> dict:
    response = await client.post("/chat", json={"query": query, "session_id": session_id})
    response.raise_for_status()
    return response.json()


async def main() -> int:
    if differing := settings_differing_from_render():
        sys.exit(f"Retrieval settings differ from render.yaml's: {differing}")
    articles = {d.doc_id: f"# {d.title}\n\n{d.text}" for d in load_documents()}
    cases = {c.id: c for c in load_golden_set()}
    conversations = [c for c in cases.values() if c.history]
    rows = []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=180) as client:
        for i, case in enumerate(conversations, 1):
            session, lived = None, []
            try:
                for turn in (t for t in case.history if t["role"] == "user"):
                    reply = await ask(client, turn["text"], session)
                    session = reply["session_id"]
                    lived += [Turn("user", turn["text"]), Turn("model", reply["answer"])]
                final = await ask(client, case.query, session)
                verdict = await judge_answer(case.query, case.reference_answer, final["answer"], history=lived,
                                             docs=[articles[s] for s in final["sources"] if s in articles])
                rows.append({"id": case.id, "split": case.split, "turns": len(lived) // 2 + 1,
                             "history_turns": final["history_turns"], "history_ok": final["history_turns"] == len(lived),
                             "correct": verdict.correct, "votes": verdict.votes, "sources": final["sources"],
                             "reasoning": verdict.reasoning[:300]})
            except Exception as error:  # noqa: BLE001 - one error shouldn't end the run
                rows.append({"id": case.id, "split": case.split, "error": str(error)[:200]})
            r = rows[-1]
            mark = "ERROR" if "error" in r else ("PASS" if r["correct"] else "FAIL")
            print(f"  {i}/{len(conversations)} [{mark}] {case.id}"
                  + ("" if "error" in r else f" ({r['turns']} turns, history_turns {r['history_turns']})"), flush=True)

        cache = []
        for case_id in CACHE_CASES:
            first = await ask(client, cases[case_id].query, None)
            again = await ask(client, cases[case_id].query, None)
            cache.append({"id": case_id, "same_answer": first["answer"] == again["answer"],
                          "first_ms": first["latency_ms"], "again_ms": again["latency_ms"]})
            print(f"  cache {case_id}: same answer {cache[-1]['same_answer']}, "
                  f"{first['latency_ms']:.0f} ms then {again['latency_ms']:.0f} ms", flush=True)

    scored = [r for r in rows if "error" not in r]
    summary = {
        "run": {"git_commit": _git_commit(), "via": "/chat, in process", "conversations": len(conversations)},
        "judge_pass_rate": sum(r["correct"] for r in scored) / len(scored) if scored else None,
        "history_ok": all(r["history_ok"] for r in scored),
        # A model call takes seconds; a cached answer comes back in milliseconds.
        "cache_ok": all(c["same_answer"] and c["again_ms"] < 200 for c in cache),
        "failed": [r["id"] for r in scored if not r["correct"]],
        "errored": [r["id"] for r in rows if "error" in r],
        "conversations": rows,
        "cache": cache,
    }
    out = SUMMARIES_DIR / f"chat_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nThrough /chat: {sum(r['correct'] for r in scored)}/{len(scored)} conversations passed; "
          f"history {'complete' if summary['history_ok'] else 'INCOMPLETE'}; cache {'ok' if summary['cache_ok'] else 'NOT OK'}")
    print(f"Summary: {out}")
    return 0 if scored and not summary["errored"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
