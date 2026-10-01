import json
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "golden_set.jsonl"

CATEGORIES = {
    "grounded",  # answer is in one doc
    "reasoning",  # apply a doc's policy to the user's specific situation
    "false_premise",  # question assumes something the docs contradict
    "multi_doc",  # a complete answer needs facts from two docs
    "unanswerable",  # in-domain, but no doc covers it: must not guess
    "out_of_scope",  # not a product support question at all
    "adversarial",  # prompt injection / social engineering
    "robustness",  # typos, other languages, vague phrasing
    "direct",  # greetings and small talk: no retrieval needed
    "multi_turn",  # a follow-up that only makes sense with the earlier turns
}
SPLITS = {"dev", "held_out"}


@dataclass
class EvalCase:
    id: str
    query: str
    category: str  # one of CATEGORIES
    expected_doc_ids: list[str]
    reference_answer: str
    # Earlier turns, scripted rather than generated so every run asks the
    # follow-up in the same conversation: [{"role": "user"|"model", "text": ...}]
    history: list[dict] = field(default_factory=list)
    # "dev": cases prompts and settings were tuned against. "held_out": cases
    # written before a change and kept out of its tuning, to check it
    # generalizes rather than fitting the dev cases.
    split: str = "dev"


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[EvalCase]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase(**json.loads(line)))
    return cases
