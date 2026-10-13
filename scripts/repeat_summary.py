"""Several eval runs of the same agent, as a mean and spread instead of one score.

    python -m scripts.repeat_summary data/eval/results/eval_A.json data/eval/results/eval_B.json ... \\
        [--splits held_out external] [--out data/eval/summaries/repeat_<name>.json]

A run given as `eval_B.json+eval_C.json` is B with the cases it has no result
for (errored on quota, say) taken from C, a top-up run of just those cases.

One run's pass rate moves from run to run: the agent's answers and the judge's
verdicts both vary. This takes runs of the same settings, keeps the cases every
run has (in the given splits, or all), and reports each split's pass count per
run, its mean and range, and every case that failed in any run with how many
runs it failed. A case that fails in every run is a real failure; one that fails
once in three is noise around the pass rate. Runs whose settings differ (agent
mode, prompts, model, retrieval settings) are refused, since they aren't
repeats of one agent.
"""

import argparse
import json
import sys
from pathlib import Path

# What makes two runs repeats of the same agent; the rest of "run" (time, commit, case count) may differ.
SAME = ("generation_model", "embedding_model", "corpus_docs", "agent_mode", "router_prompt", "direct_answer_prompt", "grounded_answer_prompt", "agent_prompt",
        "judge_prompt", "judge_max_votes", "retrieval_top_k", "retrieval_min_score", "retrieval_max_score_gap", "retrieval_expand_docs")


def summarise(runs: list[dict], names: list[str], splits: list[str] | None = None) -> dict:
    for key in SAME:
        values = {json.dumps(run["run"].get(key)) for run in runs}
        if len(values) > 1:
            raise ValueError(f"the runs differ in {key}: {sorted(values)}")
    by_run = [{r["id"]: r for r in run["results"] if not splits or r.get("split") in splits} for run in runs]
    common = set.intersection(*(set(cases) for cases in by_run))
    split_of = {case_id: by_run[0][case_id].get("split") for case_id in common}
    per_split = {}
    for split in sorted(set(split_of.values())):
        ids = [c for c in common if split_of[c] == split]
        passed = [sum(cases[c]["judge_correct"] for c in ids) for cases in by_run]
        per_split[split] = {
            "cases": len(ids),
            "passed_per_run": passed,
            "mean": round(sum(passed) / len(passed), 2),
            "min": min(passed),
            "max": max(passed),
        }
    failures = []
    for case_id in sorted(common):
        failed = sum(not cases[case_id]["judge_correct"] for cases in by_run)
        if failed:
            failures.append({"id": case_id, "split": split_of[case_id], "failed_runs": failed, "of": len(runs)})
    total = [sum(cases[c]["judge_correct"] for c in common) for cases in by_run]
    return {
        "runs": names,
        "settings": {key: runs[0]["run"].get(key) for key in SAME if key in runs[0]["run"]},
        "cases": len(common),
        "passed_per_run": total,
        "mean_pass_rate": round(sum(total) / (len(total) * len(common)), 4) if common else None,
        "by_split": per_split,
        "failures": sorted(failures, key=lambda f: (-f["failed_runs"], f["id"])),
        "missing_from_some_runs": sorted(set.union(*(set(c) for c in by_run)) - common),
    }


def load_run(spec: str) -> dict:
    """A results file, or "a.json+b.json": a's results plus b's for cases a has none for."""
    first, *top_ups = spec.split("+")
    run = json.loads(Path(first).read_text())
    have = {r["id"] for r in run["results"]}
    for path in top_ups:
        extra = json.loads(Path(path).read_text())
        for key in SAME:
            if json.dumps(extra["run"].get(key)) != json.dumps(run["run"].get(key)):
                raise ValueError(f"{path} differs from {first} in {key}")
        added = [r for r in extra["results"] if r["id"] not in have]
        run["results"] = run["results"] + added
        have |= {r["id"] for r in added}
    return run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="results files; a+b tops a up with b")
    parser.add_argument("--splits", nargs="*")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if len(args.results) < 2:
        sys.exit("Give at least two runs.")
    runs = [load_run(spec) for spec in args.results]
    summary = summarise(runs, ["+".join(Path(p).name for p in spec.split("+")) for spec in args.results], args.splits)
    print(f"{summary['cases']} cases in every run; passed per run {summary['passed_per_run']}, "
          f"mean {summary['mean_pass_rate']:.1%}")
    for split, s in summary["by_split"].items():
        print(f"  {split}: {s['passed_per_run']} of {s['cases']} (mean {s['mean']}, range {s['min']}-{s['max']})")
    for f in summary["failures"]:
        print(f"  {f['id']} ({f['split']}) failed {f['failed_runs']} of {f['of']}")
    if summary["missing_from_some_runs"]:
        print(f"Not in every run (errored or not selected): {summary['missing_from_some_runs']}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Summary: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
