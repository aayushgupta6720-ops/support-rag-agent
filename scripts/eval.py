"""Run the golden-set evaluation against the live agent.

Usage:
    python -m scripts.eval                      # every case
    python -m scripts.eval multi_turn greeting_1  # only these categories / case ids
    python -m scripts.eval held_out             # only one split ("dev" or "held_out")
    python -m scripts.eval --any-settings ...   # run even if retrieval settings differ from render.yaml's

A run uses the deployed retrieval settings (render.yaml's RETRIEVAL_* values),
or stops: on 2026-10-10 three runs went out with a local score gap of 0 against
the demo's 0.10, so their numbers didn't describe the deployed agent. Full
results go to data/eval/results/ (gitignored); a summary of each run goes to
data/eval/summaries/, which is committed.
"""

import asyncio
import json
import re
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "eval" / "results"
SUMMARIES_DIR = ROOT / "data" / "eval" / "summaries"
# Settings that change which chunks an answer sees, compared with the deployed ones.
DEPLOYED_SETTINGS = ("RETRIEVAL_MAX_SCORE_GAP", "RETRIEVAL_MIN_SCORE", "RETRIEVAL_TOP_K", "RETRIEVAL_EXPAND_DOCS")


def _setting_value(value) -> float:
    """A setting as a number, so render.yaml's strings compare with it: "0.10", "true"."""
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return float(value.lower() == "true")
    return float(value)


def settings_differing_from_render() -> dict[str, tuple[float, float]]:
    """{setting: (here, render.yaml)} for each deployed retrieval setting that differs. One that
    render.yaml doesn't set must be at its default here."""
    blueprint = (ROOT / "render.yaml").read_text()
    deployed = dict(re.findall(r'- key: (RETRIEVAL_\w+)\s+value: "?([0-9.]+|true|false)"?', blueprint))
    settings = get_settings()
    differing = {}
    for name in DEPLOYED_SETTINGS:
        here = _setting_value(getattr(settings, name.lower()))
        there = _setting_value(deployed[name]) if name in deployed else _setting_value(type(settings).model_fields[name.lower()].default)
        if here != there:
            differing[name] = (here, there)
    return differing


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
    print(f"Citation precision:  {report.citation_precision:.0%}")
    print(f"Citation recall:     {report.citation_recall:.0%}")
    spurious = report.spurious_citations()
    print(f"Cited with nothing to find: {len(spurious)}" + (f" {spurious}" if spurious else ""))
    print()
    print("By split:")
    for split, (passed, total) in sorted(report.pass_rate_by_split().items()):
        print(f"  {split:<15} {passed}/{total}")
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
        "retrieval_min_score": settings.retrieval_min_score,
        "retrieval_max_score_gap": settings.retrieval_max_score_gap,
        "retrieval_expand_docs": settings.retrieval_expand_docs,
        "corpus_docs": len(load_documents()),
        "agent_mode": settings.agent_mode,
        **({"agent_prompt": prompts.AGENT_PROMPT_VERSION, "agent_max_steps": settings.agent_max_steps}
           if settings.agent_mode == "loop" else {
            "router_prompt": prompts.ROUTER_PROMPT_VERSION,
            "direct_answer_prompt": prompts.DIRECT_ANSWER_PROMPT_VERSION,
            "grounded_answer_prompt": prompts.GROUNDED_ANSWER_PROMPT_VERSION,
        }),
        "judge_prompt": JUDGE_PROMPT_VERSION,
        "judge_max_votes": MAX_VOTES,
        "cases": len(case_ids),
        "settings_match_render": not settings_differing_from_render(),
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
        "citation_precision": report.citation_precision,
        "citation_recall": report.citation_recall,
        "spurious_citations": report.spurious_citations(),
        "cost_usd": report.cost_usd,
        "model_requests": report.model_requests,
        "agent_requests": report.agent_requests,
        "by_split": {
            split: {"passed": passed, "total": total}
            for split, (passed, total) in sorted(report.pass_rate_by_split().items())
        },
        "by_category": {
            category: {"passed": passed, "total": total}
            for category, (passed, total) in sorted(report.pass_rate_by_category().items())
        },
        "results": [
            {
                "id": r.case.id,
                "query": r.case.query,
                "category": r.case.category,
                "split": r.case.split,
                "expected_doc_ids": r.case.expected_doc_ids,
                "actual_sources": r.actual_sources,
                "retrieval_hit": r.retrieval_hit,
                "retrieval_recall": r.retrieval_recall,
                "reciprocal_rank": r.reciprocal_rank,
                "context_precision": r.context_precision,
                "search_query": r.search_query,
                "retrieved": r.retrieved,
                "agent_requests": r.agent_requests,
                "tool_calls": r.tool_calls,
                "cited_sources": r.cited_sources,
                "citation_precision": r.citation_precision,
                "citation_recall": r.citation_recall,
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


def save_summary(path: Path) -> Path:
    """The run without its answers: what produced it, the rates, what it cost,
    and which cases failed or errored. Committed, so the numbers in the README
    can be traced to a run."""
    full = json.loads(path.read_text())
    summary = {key: value for key, value in full.items() if key not in ("results", "errors")}
    summary["failed"] = [r["id"] for r in full["results"] if not r["judge_correct"]]
    summary["errored"] = [e["id"] for e in full.get("errors", [])]
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    out = SUMMARIES_DIR / path.name
    out.write_text(json.dumps(summary, indent=2) + "\n")
    return out


async def main() -> None:
    args = sys.argv[1:]
    differing = settings_differing_from_render()
    if differing and "--any-settings" not in args:
        lines = [f"  {name}: {here} here, {deployed} in render.yaml" for name, (here, deployed) in differing.items()]
        fix = " ".join(f"{name}={deployed}" for name, (_, deployed) in differing.items())
        sys.exit("Retrieval settings differ from the deployed ones:\n" + "\n".join(lines)
                 + f"\nRun with {fix}, or pass --any-settings to run anyway.")
    cases = load_golden_set()
    if selected := {arg for arg in args if not arg.startswith("--")}:
        cases = [case for case in cases if selected & {case.id, case.category, case.split}]
        if not cases:
            sys.exit(f"No case ids or categories match {sorted(selected)}")

    def progress(index: int, outcome) -> None:
        status = "ERROR" if isinstance(outcome, EvalCaseError) else "done"
        print(f"  {index + 1}/{len(cases)} {outcome.case.id}: {status}", flush=True)

    report = await run_evaluation(cases, on_case_done=progress)
    print()
    print_report(report)
    path = save_report(report, run_metadata([case.id for case in cases]))
    print(f"\nSaved results to {path}, summary to {save_summary(path)}")
    cost = report.cost_usd
    if report.results:
        print(f"The agent made {report.agent_requests} model calls, {report.agent_requests / len(report.results):.2f} a question.")
    print(f"{report.model_requests} chat-model requests; on a paid key ${cost['total']:.4f} "
          f"(agent ${cost['agent']:.4f}, judge ${cost['judge']:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
