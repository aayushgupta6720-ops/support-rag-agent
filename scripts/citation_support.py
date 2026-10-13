"""Check that each answer's cited articles actually say what the answer says.

    python -m scripts.citation_support data/eval/results/eval_20261013T090314Z.json
    python -m scripts.citation_support <results.json> --model gemma-4-26b-a4b-it

The eval scores citations only against the articles a case expects (precision
and recall): an answer could cite the right article and still claim something
it doesn't say. This gives a judge from another model family (Gemma, with its
own free-tier quota) the full text of every article an answer cites, and asks
whether each factual claim in the answer is stated there or follows directly
from it. Answers that cite nothing are left out. It writes the support rate and
every unsupported claim to data/eval/summaries/.

One Gemma call per citing answer; it re-uses the saved answers, so it spends no
agent calls. Progress is saved as it goes, so a stopped run picks up where it was.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import errors  # noqa: E402

from app.core.config import get_settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
SUMMARIES_DIR = ROOT / "data" / "eval" / "summaries"
RESULTS_DIR = ROOT / "data" / "eval" / "results"
PROMPT_VERSION = "citation_support_v2"

INSTRUCTIONS = """You check a support assistant's answer against the help-center articles it cites.

For every factual claim in the answer (a policy, a limit, a number, a step, a time frame, what is or isn't
possible), decide whether the cited articles state it or it follows directly from what they state. Wording
can differ. What the user said in the question counts as given, so applying an article's rule to it is
supported (an article's 30-day limit and "I asked 45 days ago" support "that's past the limit"). Greetings,
offers to help further, and saying that something isn't covered or that the user should contact support are
not factual claims. A claim that needs an article the answer didn't cite counts as unsupported, even if it's
true elsewhere."""

REPLY_FORMAT = (
    'Reply with only a JSON object: {"supported": true or false, '
    '"unsupported": ["each claim the cited articles don\'t support, quoted or paraphrased"]}'
)


def citation_prompt(question: str, answer: str, cited: dict[str, str]) -> str:
    articles = "\n\n".join(f"ARTICLE {doc_id}:\n{text.strip()}" for doc_id, text in cited.items())
    return (f"{INSTRUCTIONS}\n\nCITED ARTICLES:\n{articles}\n\nQUESTION:\n{question.strip()}"
            f"\n\nANSWER:\n{answer.strip()}\n\n{REPLY_FORMAT}")


def verdict_from(text: str) -> tuple[bool, list[str]] | None:
    """(supported, unsupported claims) from the judge's reply, which may come in a code fence."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("supported"), bool):
        return None
    claims = data.get("unsupported") or []
    return data["supported"], [str(c) for c in claims] if isinstance(claims, list) else [str(claims)]


def cited_articles(doc_ids: list[str]) -> dict[str, str] | None:
    """Each cited article's text, or None if one isn't a known article."""
    texts = {}
    for doc_id in dict.fromkeys(doc_ids):
        path = DOCS_DIR / f"{doc_id}.md"
        if not path.exists():
            return None
        texts[doc_id] = path.read_text(encoding="utf-8")
    return texts


def ask(client: genai.Client, model: str, prompt: str) -> str:
    for wait in (5, 15, 45, 90, None):
        try:
            return client.models.generate_content(model=model, contents=prompt).text or ""
        except errors.APIError as error:
            if wait is None or error.code not in (429, 500, 503):
                raise
            time.sleep(wait)
    return ""


def summarise(results_name: str, run: dict, model: str, rows: list[dict]) -> dict:
    judged = [r for r in rows if r["supported"] is not None]
    return {
        "results": results_name,
        "run": run,
        "prompt": PROMPT_VERSION,
        "judge": model,
        "citing_answers": len(rows),
        "unparsed": [r["id"] for r in rows if r["supported"] is None],
        "supported": sum(r["supported"] for r in judged),
        "support_rate": sum(r["supported"] for r in judged) / len(judged) if judged else None,
        "unsupported": [r for r in judged if not r["supported"]],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--model", default="gemma-4-26b-a4b-it")
    args = parser.parse_args(argv)

    run = json.loads(args.results.read_text())
    client = genai.Client(api_key=get_settings().gemini_api_key)
    stem = args.results.stem.removeprefix("eval_")
    progress = RESULTS_DIR / f"citation_support_{stem}_{args.model}_{PROMPT_VERSION}.jsonl"
    done = {}
    if progress.exists():
        for line in progress.read_text().splitlines():
            row = json.loads(line)
            done[row["id"]] = row

    citing = [r for r in run["results"] if r.get("cited_sources") and r.get("actual_answer")]
    rows = []
    for i, result in enumerate(citing, 1):
        if result["id"] in done:
            rows.append(done[result["id"]])
            continue
        cited = cited_articles(result["cited_sources"])
        prompt = None if cited is None else citation_prompt(result["query"], result["actual_answer"], cited)
        verdict = None if prompt is None else verdict_from(ask(client, args.model, prompt))
        rows.append({
            "id": result["id"], "category": result.get("category"), "split": result.get("split"),
            "cited": result["cited_sources"], "judge_correct": result.get("judge_correct"),
            "supported": None if verdict is None else verdict[0],
            "unsupported_claims": None if verdict is None else verdict[1],
        })
        with progress.open("a") as f:
            f.write(json.dumps(rows[-1]) + "\n")
        print(f"  {i}/{len(citing)} {result['id']}: {rows[-1]['supported']}", flush=True)
        time.sleep(2)

    summary = summarise(args.results.name, run["run"], args.model, rows)
    out = SUMMARIES_DIR / f"citation_support_{stem}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    judged = summary["citing_answers"] - len(summary["unparsed"])
    print(f"\n{summary['supported']}/{judged} citing answers fully supported by the articles they cite")
    for r in summary["unsupported"]:
        print(f"  {r['id']} (judge_correct={r['judge_correct']}): {r['unsupported_claims']}")
    print(f"Summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
