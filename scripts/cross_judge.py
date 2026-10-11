"""Re-grade a finished eval run's answers with a judge from another model family.

    python -m scripts.cross_judge data/eval/results/eval_20261011T080413Z.json
    python -m scripts.cross_judge <results.json> --model gemma-4-26b-a4b-it

The eval's judge is Gemini, the same family as the agent, and it re-checks only
failing verdicts. This grades every answer once more with an open-weights Gemma
model, given exactly what the eval's judge saw: the same judge_v2 instructions,
the case's history and reference, and the chunks the agent retrieved (fetched
again from the run's own search query, and checked against the run's record).
It reports how often the two judges agree, Cohen's kappa, and every
disagreement, and writes that summary to data/eval/summaries/.

Gemma has its own free-tier quota, separate from the agent's model, and re-uses
the saved answers, so it spends no agent calls: one Gemma call per case, plus
one embedding call per retrieved case.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import errors  # noqa: E402

from app.agent.history import Turn  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.judge import _judge_prompt  # noqa: E402
from app.eval.prompts import JUDGE_PROMPT_VERSION, JUDGE_SYSTEM_PROMPT  # noqa: E402
from app.rag.qdrant_store import SearchUnavailableError  # noqa: E402
from app.rag.retrieval import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SUMMARIES_DIR = ROOT / "data" / "eval" / "summaries"
RESULTS_DIR = ROOT / "data" / "eval" / "results"
REPLY_FORMAT = 'Reply with only a JSON object: {"correct": true or false, "reasoning": "one or two sentences"}'


def verdict_from(text: str) -> tuple[bool, str] | None:
    """The judge's verdict from its reply, which may come wrapped in a code fence."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("correct"), bool):
        return None
    return data["correct"], str(data.get("reasoning", ""))


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement beyond what the two judges' pass rates would give by chance."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if chance == 1 else (observed - chance) / (1 - chance)


async def retrieve_again(query: str):
    """The run's retrieval, retried through a brief Qdrant outage (one ended the first pass at case 40)."""
    for wait in (5, 30, 90, None):
        try:
            return await retrieve(query)
        except SearchUnavailableError:
            if wait is None:
                raise
            await asyncio.sleep(wait)


def ask(client: genai.Client, model: str, prompt: str) -> str:
    for wait in (5, 15, 45, 90, None):
        try:
            return client.models.generate_content(model=model, contents=prompt).text or ""
        except errors.APIError as error:
            if wait is None or error.code not in (429, 500, 503):
                raise
            time.sleep(wait)
    return ""


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    args = parser.parse_args()

    run = json.loads(args.results.read_text())
    settings = get_settings()
    if float(run["run"]["retrieval_max_score_gap"]) != settings.retrieval_max_score_gap:
        sys.exit(f"Set RETRIEVAL_MAX_SCORE_GAP={run['run']['retrieval_max_score_gap']}, as the run had it.")
    cases = {case.id: case for case in load_golden_set()}
    client = genai.Client(api_key=settings.gemini_api_key)

    # Each verdict is saved as it comes, so a run that stops part-way picks up where it was.
    progress = RESULTS_DIR / f"cross_judge_{args.results.stem.removeprefix('eval_')}_{args.model}.jsonl"
    done = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            row = json.loads(line)
            done[row["id"]] = row
    rows, mismatched = [], [r["id"] for r in done.values() if r.get("retrieval_differs")]
    for i, result in enumerate(run["results"], 1):
        if result["id"] in done:
            rows.append(done[result["id"]])
            continue
        case = cases[result["id"]]
        docs, differs = [], False
        if result.get("search_query"):
            chunks = await retrieve_again(result["search_query"])
            docs = [chunk.text for chunk in chunks]
            again = [[c.doc_id, round(c.score, 4)] for c in chunks]
            if again != [[d, round(s, 4)] for d, s in result["retrieved"]]:
                mismatched.append(result["id"])
                differs = True
        history = [Turn(**turn) for turn in case.history]
        prompt = (f"{JUDGE_SYSTEM_PROMPT}\n\n{_judge_prompt(case.query, case.reference_answer, result['actual_answer'], history, docs)}"
                  f"\n\n{REPLY_FORMAT}")
        verdict = verdict_from(ask(client, args.model, prompt))
        rows.append({"id": result["id"], "category": result["category"], "split": result.get("split"),
                     "gemini": result["judge_correct"], "gemini_first_vote": result["judge_votes"][0],
                     "other": None if verdict is None else verdict[0],
                     "other_reasoning": None if verdict is None else verdict[1], "retrieval_differs": differs})
        with progress.open("a") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        print(f"  {i}/{len(run['results'])} {result['id']}: gemini {result['judge_correct']}, "
              f"{args.model} {rows[-1]['other']}", flush=True)
        time.sleep(2)

    judged = [r for r in rows if r["other"] is not None]
    gemini, other = [r["gemini"] for r in judged], [r["other"] for r in judged]
    disagreements = [r for r in judged if r["gemini"] != r["other"]]
    summary = {
        "results": args.results.name,
        "run": run["run"],
        "judge_prompt": JUDGE_PROMPT_VERSION,
        "other_judge": args.model,
        "cases": len(rows),
        "unparsed": [r["id"] for r in rows if r["other"] is None],
        "retrieval_differs_from_run": mismatched,
        "gemini_pass_rate": sum(gemini) / len(judged),
        "other_pass_rate": sum(other) / len(judged),
        "agreement": 1 - len(disagreements) / len(judged),
        "cohen_kappa": cohen_kappa(gemini, other),
        "disagreements": disagreements,
    }
    out = SUMMARIES_DIR / f"cross_judge_{args.results.stem.removeprefix('eval_')}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nGemini judge passed {sum(gemini)}/{len(judged)}, {args.model} {sum(other)}/{len(judged)}; "
          f"agreement {summary['agreement']:.1%}, kappa {summary['cohen_kappa']:.2f}")
    for r in disagreements:
        print(f"  {r['id']}: gemini {r['gemini']}, {args.model} {r['other']}: {r['other_reasoning'][:200]}")
    if mismatched:
        print(f"Retrieval came back different from the run for {mismatched}: those judges saw different chunks.")
    print(f"Summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
