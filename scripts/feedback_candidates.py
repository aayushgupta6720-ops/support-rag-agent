"""Turn visitors' thumbs-down ratings into draft eval cases.

Usage:
    python -m scripts.feedback_candidates [--url URL]

The token comes from FEEDBACK_EXPORT_TOKEN, in the environment or .env.

Fetches every kept rating from GET /feedback/export (the live demo by
default) and writes each thumbs-down as a draft golden-set line to
data/eval/feedback_candidates.jsonl (gitignored, since it holds visitors'
questions). A draft isn't a test case yet: read the bad answer, then give
it a category, the expected doc ids and a reference answer before moving it
into data/eval/golden_set.jsonl, and delete its "feedback" field.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "feedback_candidates.jsonl"
LIVE_URL = "https://support-rag-agent.onrender.com"


def fetch(url: str, token: str) -> list[dict]:
    request = urllib.request.Request(f"{url}/feedback/export", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=90) as response:  # a cold start can take a minute
        return json.load(response)["feedback"]


def drafts(ratings: list[dict]) -> list[dict]:
    """One draft per thumbs-down question, newest first. A question rated
    down more than once becomes a single draft."""
    seen, out = set(), []
    for rating in ratings:
        if rating["rating"] != "down" or rating["question"].casefold() in seen:
            continue
        seen.add(rating["question"].casefold())
        out.append({
            "id": f"feedback_{rating['answer_id'][:8]}",
            "query": rating["question"],
            "category": "TODO",
            "expected_doc_ids": [],
            "reference_answer": "TODO",
            "split": "held_out",  # real visitors' questions: unseen while tuning
            "feedback": {"bad_answer": rating["answer"], "sources": rating["sources"],
                         "received_at": rating["received_at"]},
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=LIVE_URL, help=f"the deployed app (default {LIVE_URL})")
    args = parser.parse_args()
    token = get_settings().feedback_export_token  # the environment or .env
    if not token:
        sys.exit("Set FEEDBACK_EXPORT_TOKEN (environment or .env) to the token configured on the server.")

    ratings = fetch(args.url.rstrip("/"), token)
    lines = drafts(ratings)
    OUTPUT.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    downs = sum(r["rating"] == "down" for r in ratings)
    print(f"{len(ratings)} ratings, {downs} thumbs-down; wrote {len(lines)} drafts to {OUTPUT}")


if __name__ == "__main__":
    main()
