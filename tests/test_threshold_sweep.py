"""The offline threshold sweep: replaying a score threshold over saved scores."""

import pytest

from scripts.threshold_sweep import changed_cases, lossless_thresholds, sweep_row


def _row(id, expected, retrieved, category="grounded"):
    return {"id": id, "category": category, "expected_doc_ids": expected, "retrieved": retrieved}


ROWS = [
    # the right doc first, then two off-topic ones
    _row("refund", ["billing-refunds"], [("billing-refunds", 0.74), ("invoices", 0.63), ("api-keys", 0.62)]),
    # a near-miss ranked above the right doc
    _row("team", ["account-deletion"], [("team-members", 0.70), ("account-deletion", 0.66)]),
    # a two-doc answer whose second doc scores low
    _row("multi", ["password-reset", "two-factor-auth"], [("password-reset", 0.72), ("two-factor-auth", 0.60)]),
    # nothing to find: every doc retrieved is noise
    _row("sso", [], [("sign-in", 0.61), ("team-members", 0.58)], category="unanswerable"),
]


def test_no_threshold_matches_what_the_eval_measures():
    row = sweep_row(ROWS, 0.0)

    assert (row["hit_rate"], row["recall"]) == (1.0, 1.0)
    assert row["mrr"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert row["hit_at_1"] == pytest.approx(2 / 3)
    assert row["context_precision"] == pytest.approx((1 / 3 + 1 / 2 + 1) / 3)
    assert (row["emptied"], row["noise_docs"], row["changed"]) == (0, 2.0, [])


def test_a_threshold_trades_off_topic_docs_for_a_missed_second_doc():
    row = sweep_row(ROWS, 0.65)

    assert row["hit_rate"] == 1.0
    assert row["recall"] == pytest.approx((1 + 1 + 0.5) / 3)  # "multi" loses two-factor-auth at 0.60
    assert row["context_precision"] == pytest.approx((1 + 1 / 2 + 1) / 3)  # "refund" is left with only its doc
    assert row["noise_docs"] == 0.0  # the unanswerable case no longer sees any doc
    assert row["changed"] == ["refund", "multi", "sso"]


def test_a_case_can_be_left_with_nothing_unless_the_top_chunk_is_kept():
    rows = [_row("typo", ["billing-refunds"], [("billing-refunds", 0.55), ("api-keys", 0.50)])]

    assert (sweep_row(rows, 0.6)["emptied"], sweep_row(rows, 0.6)["hit_rate"]) == (1, 0.0)
    assert (sweep_row(rows, 0.6, keep_top=True)["emptied"], sweep_row(rows, 0.6, keep_top=True)["hit_rate"]) == (0, 1.0)


def test_the_highest_lossless_thresholds_come_from_the_weakest_expected_chunks():
    # hit rate holds while each case keeps its best expected chunk (lowest: team's 0.66);
    # recall holds while every expected doc keeps one (lowest: multi's 0.60)
    assert lossless_thresholds(ROWS) == (0.66, 0.60)


def test_only_cases_whose_docs_change_need_their_answers_rerun():
    assert changed_cases(ROWS, 0.59) == ["sso"]  # only its 0.58 doc goes
    assert changed_cases(ROWS, 0.625) == ["refund", "multi", "sso"]


def test_a_gap_follows_each_querys_best_chunk_instead_of_a_fixed_floor():
    row = sweep_row(ROWS, max_gap=0.05)

    # "team" keeps its right doc (0.04 below the near-miss) and "multi" loses
    # its second doc (0.12 below); the unanswerable case keeps both of its
    # docs, since they're within 0.05 of each other
    assert (row["hit_rate"], row["recall"]) == (1.0, pytest.approx((1 + 1 + 0.5) / 3))
    assert row["noise_docs"] == 2.0
    assert row["changed"] == ["refund", "multi"]
