"""Measure what a similarity threshold would do to retrieval, without
generating or judging any answers: a fixed floor (RETRIEVAL_MIN_SCORE) or a
maximum gap below the query's best chunk (RETRIEVAL_MAX_SCORE_GAP).

Usage:
    python -m scripts.threshold_sweep --collect    # route + search each case, save every chunk's score
    python -m scripts.threshold_sweep [FILE]       # sweep both over saved scores (latest file by default)
    python -m scripts.threshold_sweep --changed --gap 0.08 [FILE]   # case ids that setting changes

--collect makes one model call per case (the router's), where a full eval
makes three, and saves each case's search query and the score of every
chunk retrieved, with no threshold applied. A threshold can only drop
chunks, so its effect on retrieval can then be computed offline from those
scores, for any value. Answers still need an eval run, but only for the
cases whose retrieved docs the threshold changes: --changed lists them, to
pass to `python -m scripts.eval`.
"""

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import route_and_retrieve  # noqa: E402
from app.agent.history import Turn  # noqa: E402
from app.agent.prompts import ROUTER_PROMPT_VERSION  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.dataset import EvalCase, load_golden_set  # noqa: E402
from app.eval.harness import score_retrieval  # noqa: E402
from scripts.eval import RESULTS_DIR, _git_commit  # noqa: E402
from scripts.ingest import load_documents  # noqa: E402

THRESHOLDS = [0.55, 0.58, 0.60, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.70]
GAPS = [0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]


def _searches(case: EvalCase) -> bool:
    """Cases the agent should search for: the ones that expect docs, plus
    unanswerable and adversarial ones, where every doc that gets through is
    noise the answer has to ignore."""
    return bool(case.expected_doc_ids) or case.category in {"unanswerable", "adversarial"}


async def collect(cases: list[EvalCase]) -> list[dict]:
    settings = get_settings()  # record every chunk; thresholds are applied offline
    settings.retrieval_min_score = settings.retrieval_max_score_gap = 0.0
    rows = []
    for index, case in enumerate(cases):
        try:
            state = await route_and_retrieve(case.query, history=[Turn(**turn) for turn in case.history])
        except Exception as exc:  # one API failure shouldn't discard the rest
            print(f"  {index + 1}/{len(cases)} {case.id}: ERROR {type(exc).__name__}: {exc}", flush=True)
            continue
        rows.append({
            "id": case.id,
            "category": case.category,
            "split": case.split,
            "expected_doc_ids": case.expected_doc_ids,
            "search_query": state.get("search_query"),
            "retrieved": [(chunk.doc_id, chunk.score) for chunk in state.get("chunks") or []],
        })
        print(f"  {index + 1}/{len(cases)} {case.id}: {len(rows[-1]['retrieved'])} chunks", flush=True)
    return rows


def _kept_docs(retrieved: list, min_score: float = 0.0, max_gap: float = 0.0, keep_top: bool = False) -> list[str]:
    """The same filter as app.rag.retrieval.retrieve, on saved scores."""
    if max_gap and retrieved:
        min_score = max(min_score, retrieved[0][1] - max_gap)
    kept = [doc_id for doc_id, score in retrieved if score >= min_score]
    if keep_top and not kept and retrieved:
        kept = [retrieved[0][0]]
    return kept


def sweep_row(rows: list[dict], min_score: float = 0.0, max_gap: float = 0.0, keep_top: bool = False) -> dict:
    """Retrieval metrics if chunks were filtered this way. The same metrics
    as the eval, over the cases that expect docs, plus how many of those
    would be left with nothing and how many docs still reach the answer on
    cases with nothing to find."""
    def kept(r):
        return _kept_docs(r["retrieved"], min_score, max_gap, keep_top)

    expecting = [r for r in rows if r["expected_doc_ids"]]
    scores = [score_retrieval(r["expected_doc_ids"], kept(r)) for r in expecting]
    emptied = sum(bool(r["retrieved"]) and not kept(r) for r in expecting)
    noise = [len(set(kept(r))) for r in rows if not r["expected_doc_ids"] and r["retrieved"]]

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "hit_rate": mean([s.hit for s in scores]),
        "recall": mean([s.recall for s in scores]),
        "mrr": mean([s.reciprocal_rank for s in scores]),
        "hit_at_1": mean([s.reciprocal_rank == 1.0 for s in scores]),
        "context_precision": mean([s.context_precision for s in scores]),
        "emptied": emptied,
        "noise_docs": mean(noise),
        "changed": changed_cases(rows, min_score, max_gap, keep_top),
    }


def changed_cases(rows: list[dict], min_score: float = 0.0, max_gap: float = 0.0, keep_top: bool = False) -> list[str]:
    """Cases whose set of retrieved docs the filter changes: the only ones
    whose answers it can change, so the only ones an eval needs to rerun."""
    return [r["id"] for r in rows
            if set(_kept_docs(r["retrieved"], min_score, max_gap, keep_top)) != {doc_id for doc_id, _ in r["retrieved"]}]


def lossless_thresholds(rows: list[dict]) -> tuple[float, float]:
    """The highest floor that keeps hit rate at its unfiltered value, and the
    highest that keeps recall there: the lowest-scoring chunk that is some
    case's best (or only) chunk of an expected doc."""
    best_for_hit, best_for_recall = [], []
    for r in rows:
        best = {}
        for doc_id, score in r["retrieved"]:
            if doc_id in r["expected_doc_ids"]:
                best[doc_id] = max(score, best.get(doc_id, score))
        if best:
            best_for_hit.append(max(best.values()))
            best_for_recall.extend(best.values())
    return min(best_for_hit, default=0.0), min(best_for_recall, default=0.0)


def _print_table(rows: list[dict], label: str, settings: list[tuple[str, float, float]], keep_top: bool) -> None:
    print(f"{label:>9}   hit   recall   MRR   hit@1  precision  emptied  noise docs  changed cases")
    for name, min_score, max_gap in settings:
        row = sweep_row(rows, min_score, max_gap, keep_top)
        print(f"{name:>9}  {row['hit_rate']:4.0%}  {row['recall']:5.0%}   {row['mrr']:.2f}  {row['hit_at_1']:4.0%}"
              f"  {row['context_precision']:8.0%}  {row['emptied']:7d}  {row['noise_docs']:10.1f}  {len(row['changed']):13d}")
    print()


def print_sweep(rows: list[dict], keep_top: bool) -> None:
    expecting = [r for r in rows if r["expected_doc_ids"]]
    off_topic = sorted(score for r in rows for doc_id, score in r["retrieved"] if doc_id not in r["expected_doc_ids"])
    on_topic = sorted(score for r in expecting for doc_id, score in r["retrieved"] if doc_id in r["expected_doc_ids"])
    for_hit, for_recall = lossless_thresholds(rows)
    print(f"{len(rows)} cases searched, {len(expecting)} of them expect docs.")
    if on_topic and off_topic:
        print(f"Expected-doc chunks score {on_topic[0]:.3f}-{on_topic[-1]:.3f} (median {statistics.median(on_topic):.3f}); "
              f"off-topic chunks {off_topic[0]:.3f}-{off_topic[-1]:.3f} (median {statistics.median(off_topic):.3f}).")
    print(f"Highest floor with no loss: {for_hit:.3f} for hit rate, {for_recall:.3f} for recall.")
    if keep_top:
        print("The top chunk is always kept, so no case is left with nothing.")
    print()
    floors = sorted({*THRESHOLDS, round(for_hit, 3), round(for_recall, 3)})
    _print_table(rows, "floor", [("off", 0.0, 0.0)] + [(f"{t:.3f}", t, 0.0) for t in floors], keep_top)
    _print_table(rows, "gap", [("off", 0.0, 0.0)] + [(f"{g:.2f}", 0.0, g) for g in GAPS], keep_top)
    print("noise docs: distinct docs still retrieved, on average, for searched cases with nothing to find "
          "(unanswerable, and adversarial ones without expected docs).")


def _latest_scores_file() -> Path:
    files = sorted(RESULTS_DIR.glob("retrieval_*.json"))
    if not files:
        sys.exit("No saved scores yet: run with --collect first.")
    return files[-1]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", type=Path, help="saved scores (default: the latest retrieval_*.json)")
    parser.add_argument("--collect", action="store_true", help="route and search each case, and save the scores")
    parser.add_argument("--split", choices=["dev", "held_out"], help="with --collect, only this split's cases")
    parser.add_argument("--keep-top", action="store_true", help="never drop a case's top chunk")
    parser.add_argument("--changed", action="store_true", help="print the case ids --min-score/--gap change")
    parser.add_argument("--min-score", type=float, default=0.0, help="the floor to test with --changed")
    parser.add_argument("--gap", type=float, default=0.0, help="the maximum gap to test with --changed")
    args = parser.parse_args()

    if args.collect:
        cases = [case for case in load_golden_set() if _searches(case) and (not args.split or case.split == args.split)]
        rows = await collect(cases)
        settings = get_settings()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"retrieval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        run = {
            "git_commit": _git_commit(),
            "embedding_model": settings.embedding_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "router_prompt": ROUTER_PROMPT_VERSION,
            "corpus_docs": len(load_documents()),
            "cases": len(rows),
        }
        path.write_text(json.dumps({"run": run, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nSaved scores to {path}\n")
    else:
        path = args.file or _latest_scores_file()
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]

    if args.changed:
        print(" ".join(changed_cases(rows, args.min_score, args.gap, args.keep_top)))
    else:
        print_sweep(rows, args.keep_top)


if __name__ == "__main__":
    asyncio.run(main())
