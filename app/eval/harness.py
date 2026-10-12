from collections.abc import Callable
from dataclasses import dataclass, field

from app.agent.graph import run_agent
from app.agent.history import Turn
from app.eval.dataset import EvalCase
from app.core.observability import start_trace
from app.eval.judge import judge_answer


@dataclass
class RetrievalScores:
    hit: bool
    recall: float
    reciprocal_rank: float
    context_precision: float


def score_retrieval(expected_doc_ids: list[str], sources: list[str]) -> RetrievalScores:
    """Score what was retrieved (`sources`: one doc id per chunk, best first)
    against the docs a case expects. Shared by the eval and the threshold
    sweep, so both measure the same way."""
    retrieved_docs = list(dict.fromkeys(sources))
    found = [doc_id in retrieved_docs for doc_id in expected_doc_ids]
    ranks = [retrieved_docs.index(doc_id) + 1 for doc_id in expected_doc_ids if doc_id in retrieved_docs]
    relevant = sum(doc_id in expected_doc_ids for doc_id in retrieved_docs)
    return RetrievalScores(
        hit=any(found),
        recall=sum(found) / len(found),
        reciprocal_rank=1 / min(ranks) if ranks else 0.0,
        context_precision=relevant / len(retrieved_docs) if retrieved_docs else 0.0,
    )


# Trace steps that are a chat-model request (embedding calls are another quota).
# "agent" is the tool loop's call (AGENT_MODE=loop); "route" and "generate" the router's.
MODEL_STEPS = {"route", "generate", "agent", "judge"}


@dataclass
class EvalCaseResult:
    case: EvalCase
    actual_answer: str
    actual_sources: list[str]
    retrieval_hit: bool | None  # None when the case has no expected docs to check
    # Fraction of expected docs retrieved. Differs from retrieval_hit only for
    # multi-doc cases, where finding one of two docs isn't enough.
    retrieval_recall: float | None
    # 1 / rank of the first expected doc among the distinct docs retrieved
    # (0 if none was): hit rate says whether it was found, this says how high.
    reciprocal_rank: float | None
    # Share of the distinct docs retrieved that were expected: how much of
    # the context was noise.
    context_precision: float | None
    judge_correct: bool
    judge_reasoning: str
    judge_votes: list[bool]
    # What the router searched for (None if it answered without searching)
    # and each retrieved chunk's (doc id, similarity), best first: enough to
    # replay a score threshold offline.
    search_query: str | None = None
    retrieved: list[tuple[str, float]] = field(default_factory=list)
    # The docs the answer says it used. Precision: the share of them that
    # were expected (None if it cited none or the case expects none).
    # Recall: the share of expected docs it cited (None if none expected).
    cited_sources: list[str] = field(default_factory=list)
    citation_precision: float | None = None
    citation_recall: float | None = None
    # What the case cost: the agent's model calls and the judge's, from the
    # trace (the judge used to go unrecorded, though it's about a third of a run).
    agent_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    model_requests: int = 0
    # The agent's own model calls (the judge's left out), and in loop mode the
    # tools it called, in order: what a question costs the live daily budget.
    agent_requests: int = 0
    tool_calls: list[str] = field(default_factory=list)


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

    def _mean(self, field_name: str) -> float:
        """Mean of a per-case retrieval field over the cases it applies to."""
        values = [getattr(r, field_name) for r in self.results if getattr(r, field_name) is not None]
        return sum(values) / len(values) if values else 0.0

    @property
    def retrieval_hit_rate(self) -> float:
        return self._mean("retrieval_hit")

    @property
    def retrieval_recall(self) -> float:
        return self._mean("retrieval_recall")

    @property
    def mrr(self) -> float:
        return self._mean("reciprocal_rank")

    @property
    def hit_at_1(self) -> float:
        """Share of cases whose top retrieved doc was an expected one."""
        ranks = [r.reciprocal_rank for r in self.results if r.reciprocal_rank is not None]
        return sum(rank == 1.0 for rank in ranks) / len(ranks) if ranks else 0.0

    @property
    def context_precision(self) -> float:
        return self._mean("context_precision")

    @property
    def citation_precision(self) -> float:
        return self._mean("citation_precision")

    @property
    def citation_recall(self) -> float:
        return self._mean("citation_recall")

    @property
    def cost_usd(self) -> dict[str, float]:
        """What the run would have cost on a paid key: the agent's calls and the judge's."""
        agent = sum(r.agent_cost_usd for r in self.results)
        judge = sum(r.judge_cost_usd for r in self.results)
        return {"agent": round(agent, 6), "judge": round(judge, 6), "total": round(agent + judge, 6)}

    @property
    def model_requests(self) -> int:
        return sum(r.model_requests for r in self.results)

    @property
    def agent_requests(self) -> int:
        return sum(r.agent_requests for r in self.results)

    def spurious_citations(self) -> list[str]:
        """Cases with nothing to find whose answer still cited a doc."""
        return [r.case.id for r in self.results if not r.case.expected_doc_ids and r.cited_sources]

    def pass_rate_by_split(self) -> dict[str, tuple[int, int]]:
        """{split: (passed, total)}: held-out cases show whether a change
        generalizes beyond the cases it was tuned on."""
        by_split: dict[str, tuple[int, int]] = {}
        for r in self.results:
            passed, total = by_split.get(r.case.split, (0, 0))
            by_split[r.case.split] = (passed + r.judge_correct, total + 1)
        return by_split

    def pass_rate_by_category(self) -> dict[str, tuple[int, int]]:
        """{category: (passed, total)}, so one strong category can't hide a weak one."""
        by_category: dict[str, tuple[int, int]] = {}
        for r in self.results:
            passed, total = by_category.get(r.case.category, (0, 0))
            by_category[r.case.category] = (passed + r.judge_correct, total + 1)
        return by_category


async def run_case(case: EvalCase) -> EvalCaseResult:
    trace = start_trace()
    history = [Turn(**turn) for turn in case.history]
    result = await run_agent(case.query, history=history)
    actual_answer = result.get("answer", "")
    # Retrieval is scored on what the search found, one entry per chunk.
    actual_sources = result.get("retrieved_sources", result.get("sources", []))
    scores = score_retrieval(case.expected_doc_ids, actual_sources) if case.expected_doc_ids else None
    cited = list(dict.fromkeys(result.get("sources", [])))
    cited_expected = [doc_id for doc_id in cited if doc_id in case.expected_doc_ids]
    citation_precision = len(cited_expected) / len(cited) if cited and case.expected_doc_ids else None
    citation_recall = len(cited_expected) / len(case.expected_doc_ids) if case.expected_doc_ids else None

    verdict = await judge_answer(
        case.query,
        case.reference_answer,
        actual_answer,
        history=history,
        docs=[chunk.text for chunk in result.get("chunks") or []],
    )

    return EvalCaseResult(
        case=case,
        actual_answer=actual_answer,
        actual_sources=actual_sources,
        retrieval_hit=None if scores is None else scores.hit,
        retrieval_recall=None if scores is None else scores.recall,
        reciprocal_rank=None if scores is None else scores.reciprocal_rank,
        context_precision=None if scores is None else scores.context_precision,
        judge_correct=verdict.correct,
        judge_reasoning=verdict.reasoning,
        judge_votes=verdict.votes,
        search_query=result.get("search_query"),
        retrieved=[(chunk.doc_id, chunk.score) for chunk in result.get("chunks") or []],
        cited_sources=cited,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        agent_cost_usd=round(sum(s.cost_usd for s in trace.steps if s.name != "judge"), 8),
        judge_cost_usd=round(sum(s.cost_usd for s in trace.steps if s.name == "judge"), 8),
        model_requests=sum(s.name in MODEL_STEPS for s in trace.steps),
        agent_requests=sum(s.name in MODEL_STEPS - {"judge"} for s in trace.steps),
        tool_calls=list(result.get("tool_calls") or []),
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
