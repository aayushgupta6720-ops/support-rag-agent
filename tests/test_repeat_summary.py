import pytest

from scripts import repeat_summary

SETTINGS = {"generation_model": "m", "router_prompt": "router_v5", "retrieval_max_score_gap": 0.1}


def run(passes: dict, **settings) -> dict:
    return {
        "run": {**SETTINGS, **settings},
        "results": [{"id": case_id, "split": split, "judge_correct": ok} for case_id, (split, ok) in passes.items()],
    }


def test_repeats_give_a_mean_a_range_and_how_often_each_case_failed():
    runs = [
        run({"a": ("held_out", True), "b": ("held_out", False), "x": ("external", True), "d": ("dev", False)}),
        run({"a": ("held_out", True), "b": ("held_out", False), "x": ("external", False)}),
        run({"a": ("held_out", True), "b": ("held_out", True), "x": ("external", True)}),
    ]
    summary = repeat_summary.summarise(runs, ["r1", "r2", "r3"], ["held_out", "external"])
    assert summary["cases"] == 3 and summary["passed_per_run"] == [2, 1, 3]
    assert summary["mean_pass_rate"] == pytest.approx(6 / 9, abs=1e-4)
    assert summary["by_split"]["held_out"] == {"cases": 2, "passed_per_run": [1, 1, 2], "mean": 1.33, "min": 1, "max": 2}
    assert summary["failures"] == [
        {"id": "b", "split": "held_out", "failed_runs": 2, "of": 3},
        {"id": "x", "split": "external", "failed_runs": 1, "of": 3},
    ]
    assert summary["missing_from_some_runs"] == []


def test_a_case_missing_from_a_run_is_left_out_and_named():
    runs = [run({"a": ("held_out", True), "b": ("held_out", True)}), run({"a": ("held_out", True)})]
    summary = repeat_summary.summarise(runs, ["r1", "r2"])
    assert summary["cases"] == 1 and summary["missing_from_some_runs"] == ["b"]


def test_runs_of_different_agents_are_refused():
    with pytest.raises(ValueError, match="router_prompt"):
        repeat_summary.summarise([run({"a": ("dev", True)}), run({"a": ("dev", True)}, router_prompt="router_v4")], ["r1", "r2"])


def test_a_run_can_be_topped_up_with_the_cases_it_errored_on(tmp_path):
    import json

    main = tmp_path / "a.json"
    main.write_text(json.dumps(run({"x": ("held_out", True)})))
    top_up = tmp_path / "b.json"
    top_up.write_text(json.dumps(run({"x": ("held_out", False), "y": ("held_out", True)})))
    merged = repeat_summary.load_run(f"{main}+{top_up}")
    assert {r["id"]: r["judge_correct"] for r in merged["results"]} == {"x": True, "y": True}  # a's own result wins
    other = tmp_path / "c.json"
    other.write_text(json.dumps(run({"y": ("held_out", True)}, router_prompt="router_v4")))
    with pytest.raises(ValueError, match="router_prompt"):
        repeat_summary.load_run(f"{main}+{other}")
