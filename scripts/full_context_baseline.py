"""What retrieval buys: the golden set answered with every article in the prompt.

    python -m scripts.full_context_baseline                 # every case
    python -m scripts.full_context_baseline held_out multi_doc

The whole help center is 11 articles, about 8,700 characters, so it fits in one
prompt with room to spare. This skips the router and the search: each question
goes to the agent's own grounded-answer step (same prompt, same model) with all
the articles as its context, and the eval's judge grades it, seeing those
articles as it would have seen the retrieved ones. If retrieval is doing its
job, the agent should match this; where it falls short, retrieval left
something out. A summary goes to data/eval/summaries/full_context_<time>.json.

About two chat-model requests per case: the answer, and the judge.
"""

import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import prompts  # noqa: E402
from app.agent.graph import _generate  # noqa: E402
from app.agent.history import Turn  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.eval.judge import judge_answer  # noqa: E402
from app.eval.prompts import JUDGE_PROMPT_VERSION  # noqa: E402
from app.rag.documents import load_documents  # noqa: E402
from app.rag.ingest import _with_title  # noqa: E402
from app.rag.retrieval import RetrievedChunk  # noqa: E402
from scripts.eval import _git_commit  # noqa: E402

SUMMARIES_DIR = Path(__file__).resolve().parent.parent / "data" / "eval" / "summaries"


async def main() -> int:
    selected = set(sys.argv[1:])
    cases = [c for c in load_golden_set() if not selected or selected & {c.id, c.category, c.split}]
    articles = [RetrievedChunk(d.doc_id, d.title, _with_title(d.title, d.text), 1.0) for d in load_documents()]
    rows = []
    for i, case in enumerate(cases, 1):
        history = [Turn(**turn) for turn in case.history]
        try:
            out = await _generate({"query": case.query, "history": history, "chunks": articles})
            verdict = await judge_answer(case.query, case.reference_answer, out["answer"], history=history,
                                         docs=[a.text for a in articles])
            rows.append({"id": case.id, "category": case.category, "split": case.split, "correct": verdict.correct,
                         "votes": verdict.votes, "cited": out["sources"], "reasoning": verdict.reasoning})
        except Exception as error:  # noqa: BLE001 - one error shouldn't end the run
            rows.append({"id": case.id, "category": case.category, "split": case.split, "error": str(error)[:200]})
        mark = "ERROR" if "error" in rows[-1] else ("PASS" if rows[-1]["correct"] else "FAIL")
        print(f"  {i}/{len(cases)} [{mark}] {case.id}", flush=True)

    scored = [r for r in rows if "error" not in r]
    def rate(key):
        passed, total = Counter(), Counter()
        for r in scored:
            total[r[key]] += 1
            passed[r[key]] += r["correct"]
        return {k: {"passed": passed[k], "total": total[k]} for k in sorted(total)}
    settings = get_settings()
    summary = {
        "run": {"git_commit": _git_commit(), "generation_model": settings.generation_model,
                "grounded_answer_prompt": prompts.GROUNDED_ANSWER_PROMPT_VERSION, "judge_prompt": JUDGE_PROMPT_VERSION,
                "context": f"all {len(articles)} articles", "cases": len(cases)},
        "judge_pass_rate": sum(r["correct"] for r in scored) / len(scored) if scored else None,
        "by_split": rate("split"),
        "by_category": rate("category"),
        "failed": [r["id"] for r in scored if not r["correct"]],
        "errored": [r["id"] for r in rows if "error" in r],
        "results": [{k: v for k, v in r.items() if k != "reasoning"} | {"reasoning": r.get("reasoning", "")[:300]}
                    for r in rows],
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = SUMMARIES_DIR / f"full_context_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWith every article in the prompt: {sum(r['correct'] for r in scored)}/{len(scored)} "
          f"({len(rows) - len(scored)} errored); failed: {summary['failed']}")
    print(f"By split: {summary['by_split']}\nSummary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
