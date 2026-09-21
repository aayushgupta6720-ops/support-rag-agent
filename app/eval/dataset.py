import json
from dataclasses import dataclass
from pathlib import Path

GOLDEN_SET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "golden_set.jsonl"


@dataclass
class EvalCase:
    id: str
    query: str
    category: str  # "grounded" | "direct" | "unanswerable"
    expected_doc_ids: list[str]
    reference_answer: str


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[EvalCase]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(EvalCase(**json.loads(line)))
    return cases
