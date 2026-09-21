from dataclasses import dataclass

from app.agent.graph import run_agent
from app.eval.dataset import EvalCase
from app.eval.judge import judge_answer


@dataclass
class EvalCaseResult:
    case: EvalCase
    actual_answer: str
    actual_sources: list[str]
    retrieval_hit: bool | None  # None when the case has no expected docs to check
    judge_correct: bool
    judge_reasoning: str


@dataclass
class EvalReport:
    results: list[EvalCaseResult]

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


async def run_case(case: EvalCase) -> EvalCaseResult:
    result = await run_agent(case.query)
    actual_answer = result.get("answer", "")
    actual_sources = result.get("sources", [])

    retrieval_hit = None
    if case.expected_doc_ids:
        retrieval_hit = any(doc_id in actual_sources for doc_id in case.expected_doc_ids)

    verdict = await judge_answer(case.query, case.reference_answer, actual_answer)

    return EvalCaseResult(
        case=case,
        actual_answer=actual_answer,
        actual_sources=actual_sources,
        retrieval_hit=retrieval_hit,
        judge_correct=verdict.correct,
        judge_reasoning=verdict.reasoning,
    )


async def run_evaluation(cases: list[EvalCase]) -> EvalReport:
    results = [await run_case(case) for case in cases]
    return EvalReport(results=results)
