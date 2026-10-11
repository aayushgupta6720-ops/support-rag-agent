"""Answers when the help articles disagree.

    AGENT_MODE=loop python -m scripts.eval_conflicts      # or AGENT_MODE=router
    python -m scripts.eval_conflicts --no-ingest          # the collection is already built

Real help centers keep stale pages. data/eval/conflicts/docs holds three that
contradict the live articles, each dated as stale content would be: a 2024
billing FAQ (30-day annual refunds, two payment retries), an old page on
text-message sign-in codes (SMS two-factor isn't supported), and an
announcement of higher paid-plan API limits. They never go into the live
collection: this builds its own, `support_docs_conflicts`, from the live
articles plus those three, and runs the cases in data/eval/conflicts/cases.jsonl
against it. A conflict case passes when the answer says the articles disagree
and gives both, rather than silently picking one; a control case, where
nothing disagrees, passes when the answer doesn't invent a conflict. The
eval's judge grades each, seeing the retrieved passages. A summary goes to
data/eval/summaries/conflicts_<mode>_<time>.json.

About 3 chat-model requests per case (the agent, then the judge), plus the
embeddings for the collection the first time.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["QDRANT_COLLECTION"] = "support_docs_conflicts"  # before the settings load
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.agent.loop as loop  # noqa: E402
from app.agent.graph import run_agent  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.judge import judge_answer  # noqa: E402
from app.rag.documents import load_documents  # noqa: E402
from app.rag.ingest import ingest_documents  # noqa: E402
from scripts.eval import _git_commit, settings_differing_from_render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFLICTS = ROOT / "data" / "eval" / "conflicts"
SUMMARIES_DIR = ROOT / "data" / "eval" / "summaries"


async def main() -> int:
    settings = get_settings()
    if settings.qdrant_collection == "support_docs":
        sys.exit("Refusing to put conflicting articles into the live collection.")
    if differing := settings_differing_from_render():
        sys.exit(f"Retrieval settings differ from render.yaml's: {differing}")
    corpus = load_documents() + load_documents(CONFLICTS / "docs")
    if "--no-ingest" not in sys.argv:
        chunks = await ingest_documents(corpus, prune_missing=True)
        print(f"Collection {settings.qdrant_collection}: {len(corpus)} articles, {chunks} chunks.")
    # read_article should find the conflicting articles too, as it would in a help center that had them.
    articles = {doc.doc_id: doc for doc in corpus}
    loop._articles = lambda: articles

    cases = [json.loads(line) for line in (CONFLICTS / "cases.jsonl").read_text().splitlines() if line.strip()]
    rows = []
    for i, case in enumerate(cases, 1):
        try:
            result = await run_agent(case["query"])
            chunks = result.get("chunks") or []
            verdict = await judge_answer(case["query"], case["reference_answer"], result["answer"],
                                         docs=[chunk.text for chunk in chunks])
            seen = list(dict.fromkeys(chunk.doc_id for chunk in chunks))
            rows.append({"id": case["id"], "category": case["category"], "correct": verdict.correct,
                         "votes": verdict.votes, "saw_both": set(case["expected_doc_ids"]) <= set(seen),
                         "retrieved": seen, "cited": result["sources"], "tool_calls": result.get("tool_calls"),
                         "answer": result["answer"], "reasoning": verdict.reasoning})
        except Exception as error:  # noqa: BLE001 - one error shouldn't end the run
            rows.append({"id": case["id"], "category": case["category"], "error": str(error)[:200]})
        r = rows[-1]
        mark = "ERROR" if "error" in r else ("PASS" if r["correct"] else "FAIL")
        print(f"  {i}/{len(cases)} [{mark}] {case['id']}" + ("" if "error" in r else f" (saw both: {r['saw_both']})"), flush=True)

    scored = [r for r in rows if "error" not in r]
    def rate(category):
        group = [r for r in scored if r["category"] == category]
        return {"passed": sum(r["correct"] for r in group), "total": len(group)}
    summary = {
        "run": {"git_commit": _git_commit(), "agent_mode": settings.agent_mode, "generation_model": settings.generation_model,
                "collection": settings.qdrant_collection, "retrieval_max_score_gap": settings.retrieval_max_score_gap,
                "retrieval_expand_docs": settings.retrieval_expand_docs, "cases": len(cases)},
        "conflicting": rate("conflicting"),
        "controls": rate("conflict_control"),
        "failed": [r["id"] for r in scored if not r["correct"]],
        "errored": [r["id"] for r in rows if "error" in r],
        "results": rows,
    }
    out = SUMMARIES_DIR / f"conflicts_{settings.agent_mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n{settings.agent_mode}: conflicts {summary['conflicting']['passed']}/{summary['conflicting']['total']}, "
          f"controls {summary['controls']['passed']}/{summary['controls']['total']}; summary {out}")
    return 0 if not summary["errored"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
