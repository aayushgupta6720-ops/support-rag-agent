from collections.abc import Callable
from dataclasses import dataclass, field

from app.agent.graph import run_agent
from app.eval.dataset import EvalCase
from app.eval.judge import judge_answer


@dataclass
class EvalCaseResult:
    case: EvalCase
    actual_answer: str
    actual_sources: list[str]
    retrieval_hit: bool | None  # None when the case has no expected docs to check
    # Fraction of expected docs retrieved. Differs from retrieval_hit only for
    # multi-doc cases, where finding one of two docs isn't enough.
    retrieval_recall: float | None
    judge_correct: bool
    judge_reasoning: str


@dataclass
class EvalCaseError:
    case: EvalCase
    error: str


@dataclass
class EvalReport:
    results: list[EvalCaseResult]
    # Cases that raised (e.g. a Gemini 503) instead of producing an answer.
    # Kept out of every rate below: they measure the API, not the agent.
    errors: list[EvalCaseError] = field(default_factory=list)

    @property
    def judge_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.judge_correct for r in self.results) / len(self.results)

    @property
    def retrieval_hit_rate(self) -> float:
        applicable = [r for r in self.results if r.retrieval_hit is not None]
        if not applicable:
            return 0.0
        return sum(r.retrieval_hit for r in applicable) / len(applicable)

    @property
    def retrieval_recall(self) -> float:
        applicable = [r for r in self.results if r.retrieval_recall is not None]
        if not applicable:
            return 0.0
        return sum(r.retrieval_recall for r in applicable) / len(applicable)

    def pass_rate_by_category(self) -> dict[str, tuple[int, int]]:
        """{category: (passed, total)}, so one strong category can't hide a weak one."""
        by_category: dict[str, tuple[int, int]] = {}
        for r in self.results:
            passed, total = by_category.get(r.case.category, (0, 0))
            by_category[r.case.category] = (passed + r.judge_correct, total + 1)
        return by_category


async def run_case(case: EvalCase) -> EvalCaseResult:
    result = await run_agent(case.query)
    actual_answer = result.get("answer", "")
    actual_sources = result.get("sources", [])

    retrieval_hit = None
    retrieval_recall = None
    if case.expected_doc_ids:
        found = [doc_id in actual_sources for doc_id in case.expected_doc_ids]
        retrieval_hit = any(found)
        retrieval_recall = sum(found) / len(found)

    verdict = await judge_answer(case.query, case.reference_answer, actual_answer)

    return EvalCaseResult(
        case=case,
        actual_answer=actual_answer,
        actual_sources=actual_sources,
        retrieval_hit=retrieval_hit,
        retrieval_recall=retrieval_recall,
        judge_correct=verdict.correct,
        judge_reasoning=verdict.reasoning,
    )


async def run_evaluation(
    cases: list[EvalCase],
    on_case_done: Callable[[int, EvalCaseResult | EvalCaseError], None] | None = None,
) -> EvalReport:
    """Run every case, recording a case that raises as an error instead of
    aborting: one transient API failure shouldn't discard a 30-minute run."""
    report = EvalReport(results=[])
    for index, case in enumerate(cases):
        try:
            outcome = await run_case(case)
            report.results.append(outcome)
        except Exception as exc:
            outcome = EvalCaseError(case=case, error=f"{type(exc).__name__}: {exc}")
            report.errors.append(outcome)
        if on_case_done:
            on_case_done(index, outcome)
    return report
