"""Run the golden-set evaluation against the live agent.

Usage:
    python -m scripts.eval                      # every case
    python -m scripts.eval multi_turn greeting_1  # only these categories / case ids
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import prompts  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.harness import EvalCaseError, EvalReport, run_evaluation  # noqa: E402
from app.eval.judge import MAX_VOTES  # noqa: E402
from app.eval.prompts import JUDGE_PROMPT_VERSION  # noqa: E402
from scripts.ingest import load_documents  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "results"


def print_report(report: EvalReport) -> None:
    for result in report.results:
        judge_mark = "PASS" if result.judge_correct else "FAIL"
        retrieval_mark = (
            "-" if result.retrieval_hit is None else ("hit" if result.retrieval_hit else "MISS")
        )
        votes = "" if len(result.judge_votes) == 1 else f", judge votes: {result.judge_votes}"
        print(f"[{judge_mark}] {result.case.id} (retrieval: {retrieval_mark}{votes})")
        if not result.judge_correct:
            print(f"    query:    {result.case.query}")
            print(f"    answer:   {result.actual_answer}")
            print(f"    reason:   {result.judge_reasoning}")
    for error in report.errors:
        print(f"[ERROR] {error.case.id}: {error.error}")

    print()
    print(f"Judge pass rate:     {report.judge_pass_rate:.0%} ({len(report.results)} cases)")
    if report.errors:
        print(f"Errored (unscored):  {len(report.errors)} cases")
    print(f"Retrieval hit rate:  {report.retrieval_hit_rate:.0%}")
    print(f"Retrieval recall:    {report.retrieval_recall:.0%}")
    print(f"MRR:                 {report.mrr:.2f}")
    print(f"Hit@1:               {report.hit_at_1:.0%}")
    print(f"Context precision:   {report.context_precision:.0%}")
    print()
    print("By category:")
    for category, (passed, total) in sorted(report.pass_rate_by_category().items()):
        print(f"  {category:<15} {passed}/{total}")


def _git_commit() -> str | None:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return commit.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def run_metadata(case_ids: list[str]) -> dict:
    """What produced this run, so its numbers can be compared with others."""
    settings = get_settings()
    return {
        "git_commit": _git_commit(),
        "generation_model": settings.generation_model,
        "embedding_model": settings.embedding_model,
        "retrieval_top_k": settings.retrieval_top_k,
        "corpus_docs": len(load_documents()),
        "router_prompt": prompts.ROUTER_PROMPT_VERSION,
        "direct_answer_prompt": prompts.DIRECT_ANSWER_PROMPT_VERSION,
        "grounded_answer_prompt": prompts.GROUNDED_ANSWER_PROMPT_VERSION,
        "judge_prompt": JUDGE_PROMPT_VERSION,
        "judge_max_votes": MAX_VOTES,
        "cases": len(case_ids),
    }


def save_report(report: EvalReport, metadata: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"eval_{timestamp}.json"

    payload = {
        "run": metadata,
        "judge_pass_rate": report.judge_pass_rate,
        "retrieval_hit_rate": report.retrieval_hit_rate,
        "retrieval_recall": report.retrieval_recall,
        "mrr": report.mrr,
        "hit_at_1": report.hit_at_1,
        "context_precision": report.context_precision,
        "by_category": {
            category: {"passed": passed, "total": total}
            for category, (passed, total) in sorted(report.pass_rate_by_category().items())
        },
        "results": [
            {
                "id": r.case.id,
                "query": r.case.query,
                "category": r.case.category,
                "expected_doc_ids": r.case.expected_doc_ids,
                "actual_sources": r.actual_sources,
                "retrieval_hit": r.retrieval_hit,
                "retrieval_recall": r.retrieval_recall,
                "reciprocal_rank": r.reciprocal_rank,
                "context_precision": r.context_precision,
                "actual_answer": r.actual_answer,
                "judge_correct": r.judge_correct,
                "judge_votes": r.judge_votes,
                "judge_reasoning": r.judge_reasoning,
            }
            for r in report.results
        ],
        "errors": [
            {"id": e.case.id, "category": e.case.category, "error": e.error}
            for e in report.errors
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


async def main() -> None:
    cases = load_golden_set()
    if selected := set(sys.argv[1:]):
        cases = [case for case in cases if case.id in selected or case.category in selected]
        if not cases:
            sys.exit(f"No case ids or categories match {sorted(selected)}")

    def progress(index: int, outcome) -> None:
        status = "ERROR" if isinstance(outcome, EvalCaseError) else "done"
        print(f"  {index + 1}/{len(cases)} {outcome.case.id}: {status}", flush=True)

    report = await run_evaluation(cases, on_case_done=progress)
    print()
    print_report(report)
    path = save_report(report, run_metadata([case.id for case in cases]))
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    asyncio.run(main())
