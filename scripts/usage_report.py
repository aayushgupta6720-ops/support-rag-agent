"""Latency and cost of real chat requests, from the app's logs or an eval run.

Usage:
    python -m scripts.usage_report LOG_FILE [--json OUT]
    python -m scripts.usage_report data/eval/results/eval_<timestamp>.json [--json OUT]

LOG_FILE holds the app's log lines (Render's log export, or `docker logs`):
one JSON object per line, with any timestamp prefix before the "{". Every
"chat_call" event counts; other lines are skipped. The report gives the
request count, the share served from the answer cache, latency percentiles
(all requests, and those the agent answered), each traced step's latency,
tokens and cost per answered question, and the cost of 1,000 questions at
this cache-hit rate. `--json` also writes it as a summary file.

Every call already logs its latency, tokens and cost (app/api/routes.py); this
is what reads them back. The free plan keeps only a few days of logs, so for a
larger sample give it an eval run's results instead: every case records how
long the agent took and what it cost (the judge's share left out), so the
report is the same, with no cache in play.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.observability import percentiles  # noqa: E402


def chat_calls(lines) -> list[dict]:
    """The chat_call events among the log lines."""
    calls = []
    for line in lines:
        start = line.find("{")
        if start == -1:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "chat_call":
            calls.append(event)
    return calls


def report(calls: list[dict]) -> dict:
    ok = [c for c in calls if "error" not in c]
    answered = [c for c in ok if not c.get("cache_hit")]
    steps = defaultdict(list)
    for call in answered:
        for step in call.get("steps", []):
            steps[step["name"]].append(step.get("latency_ms"))
    cost = [c.get("total_cost_usd", 0.0) for c in answered]
    tokens = [c.get("total_tokens", 0) for c in answered]
    hit_rate = (len(ok) - len(answered)) / len(ok) if ok else 0.0
    per_question = sum(cost) / len(cost) if cost else 0.0
    return {
        "requests": len(calls),
        "errors": len(calls) - len(ok),
        "cache_hit_rate": round(hit_rate, 3),
        "latency_ms": {
            "all": percentiles(c.get("latency_ms") for c in ok),
            "answered": percentiles(c.get("latency_ms") for c in answered),
            "cached": percentiles(c.get("latency_ms") for c in ok if c.get("cache_hit")),
        },
        "steps_ms": {name: percentiles(values) for name, values in sorted(steps.items())},
        "tokens_per_answered": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "cost_usd_per_answered": round(per_question, 6),
        # A cached answer makes no model call, so it costs nothing.
        "cost_usd_per_1000_questions": round(1000 * per_question * (1 - hit_rate), 4),
    }


def eval_calls(run: dict) -> list[dict]:
    """An eval run's cases as chat_call-shaped events: answered by the agent, never cached."""
    calls = []
    for result in run.get("results", []):
        if not result.get("agent_latency_ms"):
            continue  # a run from before cases recorded their latency
        calls.append({
            "event": "chat_call",
            "cache_hit": False,
            "latency_ms": result["agent_latency_ms"],
            "total_cost_usd": result.get("agent_cost_usd", 0.0),
            "steps": [{"name": name, "latency_ms": ms} for name, ms in result.get("agent_steps_ms", {}).items()
                      if name != "judge"],
        })
    for error in run.get("errors", []):
        calls.append({"event": "chat_call", "error": error.get("error", "")})
    return calls


def _print(summary: dict) -> None:
    lat = summary["latency_ms"]
    print(f"{summary['requests']} chat requests, {summary['errors']} errors, "
          f"{summary['cache_hit_rate']:.0%} from the answer cache")
    for label, key in (("all", "all"), ("answered by the agent", "answered"), ("from the cache", "cached")):
        p = lat[key]
        if p["n"]:
            print(f"  latency, {label} (n={p['n']}): p50 {p['p50']:.0f} ms, p95 {p['p95']:.0f} ms, max {p['max']:.0f} ms")
    for name, p in summary["steps_ms"].items():
        print(f"    {name}: p50 {p['p50']:.0f} ms, p95 {p['p95']:.0f} ms (n={p['n']})")
    print(f"  {summary['tokens_per_answered']:.0f} tokens and ${summary['cost_usd_per_answered']:.5f} per answered "
          f"question; ${summary['cost_usd_per_1000_questions']:.2f} per 1,000 questions at this cache-hit rate")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--json", type=Path, help="also write the summary here")
    args = parser.parse_args(argv)
    text = args.log_file.read_text(encoding="utf-8")
    try:
        run = json.loads(text)
    except json.JSONDecodeError:
        run = None
    calls = eval_calls(run) if isinstance(run, dict) and "results" in run else chat_calls(text.splitlines())
    if not calls:
        print("No chat_call events in that file.", file=sys.stderr)
        return 1
    summary = report(calls)
    _print(summary)
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
