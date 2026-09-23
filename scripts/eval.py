"""Run the golden-set evaluation against the live agent.

Usage:
    python -m scripts.eval
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.harness import EvalCaseError, EvalReport, run_evaluation  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "results"


def print_report(report: EvalReport) -> None:
    for result in report.results:
        judge_mark = "PASS" if result.judge_correct else "FAIL"
        retrieval_mark = (
            "-" if result.retrieval_hit is None else ("hit" if result.retrieval_hit else "MISS")
        )
        print(f"[{judge_mark}] {result.case.id} (retrieval: {retrieval_mark})")
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
    print()
    print("By category:")
    for category, (passed, total) in sorted(report.pass_rate_by_category().items()):
        print(f"  {category:<15} {passed}/{total}")


def save_report(report: EvalReport) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"eval_{timestamp}.json"

    payload = {
        "judge_pass_rate": report.judge_pass_rate,
        "retrieval_hit_rate": report.retrieval_hit_rate,
        "retrieval_recall": report.retrieval_recall,
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
                "actual_answer": r.actual_answer,
                "judge_correct": r.judge_correct,
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

    def progress(index: int, outcome) -> None:
        status = "ERROR" if isinstance(outcome, EvalCaseError) else "done"
        print(f"  {index + 1}/{len(cases)} {outcome.case.id}: {status}", flush=True)

    report = await run_evaluation(cases, on_case_done=progress)
    print()
    print_report(report)
    path = save_report(report)
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    asyncio.run(main())
