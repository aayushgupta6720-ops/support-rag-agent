import pytest

import app.eval.harness as harness
from app.eval.dataset import CATEGORIES, EvalCase, load_golden_set
from app.eval.judge import JudgeVerdict
from scripts.ingest import load_documents


def _case(id="c", category="grounded", expected=("password-reset",)):
    return EvalCase(id=id, query="q", category=category, expected_doc_ids=list(expected),
                    reference_answer="ref")


@pytest.fixture
def fake_agent(monkeypatch):
    """Set `.sources` / `.correct` to control what the agent and judge return."""

    class Fake:
        sources: list[str] = []
        correct = True

    fake = Fake()

    async def run_agent(query):
        return {"answer": "an answer", "sources": fake.sources}

    async def judge_answer(query, reference, actual):
        return JudgeVerdict(correct=fake.correct, reasoning="because")

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
