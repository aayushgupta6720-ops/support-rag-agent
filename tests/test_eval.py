import re

import pytest

import app.eval.harness as harness
import app.eval.judge as judge
from app.agent.history import Turn
from app.eval.dataset import CATEGORIES, SPLITS, EvalCase, load_golden_set
from app.eval.judge import JudgeResult, JudgeVerdict
from scripts.ingest import load_documents
from tests.fakes import FakeGenerate, chunk, model_response


def _case(id="c", category="grounded", expected=("password-reset",)):
    return EvalCase(id=id, query="q", category=category, expected_doc_ids=list(expected),
                    reference_answer="ref")


@pytest.fixture
def fake_agent(monkeypatch):
    """Set `.sources` / `.correct` to control what the agent and judge return."""

    class Fake:
        sources: list[str] = []
        correct = True
        judged: list[dict] = []

    fake = Fake()

    async def run_agent(query, history=None):
        return {"answer": "an answer", "sources": fake.sources, "chunks": [chunk(d, f"text of {d}") for d in fake.sources]}

    async def judge_answer(query, reference, actual, history=None, docs=None):
        fake.judged.append({"history": history, "docs": docs})
        return JudgeResult(correct=fake.correct, reasoning="because", votes=[fake.correct])

    monkeypatch.setattr(harness, "run_agent", run_agent)
    monkeypatch.setattr(harness, "judge_answer", judge_answer)
    return fake


# --- golden set integrity: a typo here would silently make a case unpassable


def test_golden_set_is_large_enough_and_ids_are_unique():
    cases = load_golden_set()
    assert len(cases) >= 50
    assert len({c.id for c in cases}) == len(cases)


def test_golden_set_uses_known_categories_and_covers_all_of_them():
    assert {c.category for c in load_golden_set()} == CATEGORIES


def test_every_case_is_in_a_known_split_and_held_out_ones_stand_alone():
    cases = load_golden_set()
    assert {c.split for c in cases} == SPLITS
    held_out = [c for c in cases if c.split == "held_out"]
    assert len(held_out) >= 10 and all(not c.history for c in held_out)


def test_expected_doc_ids_exist_in_the_corpus():
    doc_ids = {d.doc_id for d in load_documents()}
    for case in load_golden_set():
        assert set(case.expected_doc_ids) <= doc_ids, case.id


def test_category_shapes():
    for case in load_golden_set():
        if case.category == "multi_doc":
            assert len(case.expected_doc_ids) >= 2, case.id
        if case.category in {"unanswerable", "out_of_scope", "direct"}:
            assert case.expected_doc_ids == [], case.id
        # a follow-up needs a conversation to follow: user first, turns
        # alternating, the model's reply last; other cases stand alone
        roles = [turn["role"] for turn in case.history]
        if case.category == "multi_turn":
            assert roles and roles == ["user", "model"] * (len(roles) // 2), case.id
        else:
            assert roles == [], case.id


# Each unanswerable case stays unanswerable only while no doc covers it; a
# pattern here catches a new doc that quietly answers one.
_UNANSWERABLE_PATTERNS = {
    "unanswerable_dark_mode": r"dark mode|\btheme",
    "unanswerable_pricing": r"[$€£]\s?\d|\bprice[sd]?\b|\bcosts? \d",
    "unanswerable_sso": r"\bsso\b|saml|single sign-on",
    "unanswerable_phone_support": r"phone (number|support)|\bcall us\b|\+\d",
    "unanswerable_team_seats": r"\bseats?\b|up to \d+ (members|users|people)|\d+ (members|users) (per|max)",
    "unanswerable_upload_size": r"\b\d+\s?(kb|mb|gb)\b|file size|upload limit",
    "ho_unanswerable_android": r"android|iphone|\bios\b|mobile app",
    "h2_unanswerable_currency": r"\beuros?\b|currenc|\busd\b|\bgbp\b|dollar",
}


def test_no_doc_answers_an_unanswerable_case():
    unanswerable = {c.id for c in load_golden_set() if c.category == "unanswerable"}
    assert set(_UNANSWERABLE_PATTERNS) == unanswerable
    for doc in load_documents():
        for case_id, pattern in _UNANSWERABLE_PATTERNS.items():
            assert not re.search(pattern, f"{doc.title}\n{doc.text}", re.IGNORECASE), (doc.doc_id, case_id)


# --- harness scoring


async def test_multi_doc_partial_retrieval_is_a_hit_but_half_recall(fake_agent):
    fake_agent.sources = ["billing-refunds", "billing-refunds"]

    result = await harness.run_case(_case(expected=("account-deletion", "billing-refunds")))

    assert result.retrieval_hit is True
    assert result.retrieval_recall == 0.5


async def test_cases_without_expected_docs_are_left_out_of_retrieval_metrics(fake_agent):
    fake_agent.sources = ["password-reset"]
    report = await harness.run_evaluation([
        _case(id="a", expected=("password-reset",)),
        _case(id="b", category="unanswerable", expected=()),
    ])

    assert report.results[1].retrieval_hit is None
    assert report.results[1].retrieval_recall is None
    assert report.retrieval_hit_rate == 1.0
    assert report.retrieval_recall == 1.0


async def test_pass_rate_overall_and_by_category(fake_agent):
    results = []
    for id, category, correct in [("a", "grounded", True), ("b", "grounded", False),
                                  ("c", "adversarial", True)]:
        fake_agent.correct = correct
        results.append(await harness.run_case(_case(id=id, category=category)))
    report = harness.EvalReport(results=results)

    assert report.judge_pass_rate == pytest.approx(2 / 3)
    assert report.pass_rate_by_category() == {"grounded": (1, 2), "adversarial": (1, 1)}


def test_empty_report_has_zero_rates():
    report = harness.EvalReport(results=[])
    assert (report.judge_pass_rate, report.retrieval_hit_rate, report.retrieval_recall) == (0.0, 0.0, 0.0)
    assert (report.mrr, report.hit_at_1, report.context_precision) == (0.0, 0.0, 0.0)


async def test_rank_and_precision_count_distinct_docs_in_retrieval_order(fake_agent):
    # one entry per chunk: two chunks of the same doc are one doc for ranking
    fake_agent.sources = ["api-keys", "api-keys", "api-rate-limits", "billing-refunds"]

    result = await harness.run_case(_case(expected=("api-rate-limits",)))

    assert result.retrieval_hit is True
    assert result.reciprocal_rank == 0.5  # second distinct doc, not third chunk
    assert result.context_precision == pytest.approx(1 / 3)


async def test_a_case_records_the_search_query_and_each_chunks_score(fake_agent, monkeypatch):
    async def run_agent(query, history=None):
        return {"answer": "a", "sources": ["billing-refunds", "api-keys"], "search_query": "refund policy",
                "chunks": [chunk("billing-refunds", score=0.74), chunk("api-keys", score=0.61)]}

    monkeypatch.setattr(harness, "run_agent", run_agent)

    result = await harness.run_case(_case(expected=("billing-refunds",)))

    # enough to replay any score threshold later without calling the model
    assert result.search_query == "refund policy"
    assert result.retrieved == [("billing-refunds", 0.74), ("api-keys", 0.61)]


async def test_mrr_hit_at_1_and_precision_over_the_cases_they_apply_to(fake_agent):
    results = []
    for id, sources in [("a", ["password-reset"]), ("b", ["api-keys", "password-reset"]), ("c", ["api-keys"])]:
        fake_agent.sources = sources
        results.append(await harness.run_case(_case(id=id, expected=("password-reset",))))
    fake_agent.sources = []
    results.append(await harness.run_case(_case(id="d", category="direct", expected=())))
    report = harness.EvalReport(results=results)

    assert report.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.hit_at_1 == pytest.approx(1 / 3)
    assert report.context_precision == pytest.approx((1 + 0.5 + 0) / 3)


async def test_a_multi_turn_case_runs_in_its_conversation_and_the_judge_sees_it(fake_agent, monkeypatch):
    seen = []

    async def run_agent(query, history=None):
        seen.append(history)
        return {"answer": "14 days", "sources": ["billing-refunds"], "chunks": [chunk("billing-refunds", "Annual: 14 days.")]}

    monkeypatch.setattr(harness, "run_agent", run_agent)
    case = _case(category="multi_turn", expected=("billing-refunds",))
    case.history = [{"role": "user", "text": "Refund on monthly?"}, {"role": "model", "text": "Within 7 days."}]

    await harness.run_case(case)

    conversation = [Turn("user", "Refund on monthly?"), Turn("model", "Within 7 days.")]
    assert seen == [conversation]
    assert fake_agent.judged == [{"history": conversation, "docs": ["Annual: 14 days."]}]


# --- the judge


def _votes(*verdicts):
    return FakeGenerate([model_response(parsed=JudgeVerdict(correct=v, reasoning=f"vote {i}"))
                         for i, v in enumerate(verdicts)])


@pytest.mark.parametrize(("verdicts", "correct", "calls"), [
    ((True,), True, 1),                # a pass isn't rechecked
    ((False, False), False, 2),        # two fails: the majority, no third vote
    ((False, True, True), True, 3),    # a lone false fail is outvoted
    ((False, True, False), False, 3),
])
async def test_a_failing_verdict_is_rejudged_and_the_majority_kept(monkeypatch, verdicts, correct, calls):
    fake = _votes(*verdicts)
    monkeypatch.setattr(judge, "generate", fake)

    result = await judge.judge_answer("q", "ref", "answer")

    assert (result.correct, len(fake.calls), result.votes) == (correct, calls, list(verdicts[:calls]))
    assert result.reasoning == f"vote {list(verdicts).index(correct)}"  # from the winning side


async def test_the_judge_is_shown_the_conversation_and_the_retrieved_docs(monkeypatch):
    fake = _votes(True)
    monkeypatch.setattr(judge, "generate", fake)

    await judge.judge_answer("What about annual?", "14 days", "Within 14 days.",
                             history=[Turn("user", "Refunds on monthly?"), Turn("model", "7 days.")],
                             docs=["Annual plans: 14 days."])

    prompt = fake.calls[0]["prompt"]
    assert "User: Refunds on monthly?\nAgent: 7 days." in prompt
    assert "Annual plans: 14 days." in prompt
    assert prompt.endswith("Question: What about annual?\n\nReference: 14 days\n\nAgent answer: Within 14 days.")


async def test_a_failing_case_is_recorded_and_the_run_continues(fake_agent, monkeypatch):
    real_run_case = harness.run_case

    async def flaky_run_case(case):
        if case.id == "b":
            raise RuntimeError("503 UNAVAILABLE")
        return await real_run_case(case)

    monkeypatch.setattr(harness, "run_case", flaky_run_case)
    seen = []

    report = await harness.run_evaluation(
        [_case(id="a"), _case(id="b"), _case(id="c")],
        on_case_done=lambda i, outcome: seen.append((i, outcome.case.id)),
    )

    assert [r.case.id for r in report.results] == ["a", "c"]
    [error] = report.errors
    assert error.case.id == "b" and error.error == "RuntimeError: 503 UNAVAILABLE"
    # errored cases don't count as failures: the rate is over scored cases only
    assert report.judge_pass_rate == 1.0
    assert seen == [(0, "a"), (1, "b"), (2, "c")]


async def test_citations_are_scored_against_the_expected_docs(fake_agent, monkeypatch):
    async def run_agent(query, history=None):
        return {"answer": "a", "sources": ["billing-refunds", "invoices-and-receipts"],
                "retrieved_sources": ["billing-refunds", "invoices-and-receipts", "api-keys"],
                "chunks": [chunk("billing-refunds"), chunk("invoices-and-receipts"), chunk("api-keys")]}

    monkeypatch.setattr(harness, "run_agent", run_agent)

    result = await harness.run_case(_case(expected=("billing-refunds", "account-deletion")))

    # retrieval is still scored on what the search found
    assert result.actual_sources == ["billing-refunds", "invoices-and-receipts", "api-keys"]
    assert result.context_precision == pytest.approx(1 / 3)
    assert result.cited_sources == ["billing-refunds", "invoices-and-receipts"]
    assert (result.citation_precision, result.citation_recall) == (0.5, 0.5)


async def test_an_answer_with_nothing_to_find_should_cite_nothing(fake_agent, monkeypatch):
    async def run_agent(query, history=None):
        return {"answer": "I don't have that information.", "sources": ["team-members"], "retrieved_sources": ["team-members"]}

    monkeypatch.setattr(harness, "run_agent", run_agent)
    report = harness.EvalReport(results=[
        await harness.run_case(_case(id="seats", category="unanswerable", expected=())),
    ])

    assert report.results[0].citation_precision is None and report.results[0].citation_recall is None
    assert report.spurious_citations() == ["seats"]
