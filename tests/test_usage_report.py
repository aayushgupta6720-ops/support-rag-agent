import json

import pytest

from scripts import usage_report


def _line(**event) -> str:
    return "2026-10-13T09:00:00Z " + json.dumps({"event": "chat_call", **event})


ANSWERED = [
    _line(cache_hit=False, latency_ms=1000 + 100 * i, total_tokens=2000, total_cost_usd=0.0004,
          steps=[{"name": "router", "latency_ms": 300}, {"name": "generate", "latency_ms": 600 + 100 * i}])
    for i in range(10)
]
CACHED = [_line(cache_hit=True, latency_ms=20, total_tokens=0, total_cost_usd=0.0) for _ in range(10)]
OTHER = ['{"event": "state_store", "backend": "redis"}', "INFO:     Started server process", "{not json"]


def test_only_chat_calls_count_and_their_prefix_is_ignored():
    calls = usage_report.chat_calls(ANSWERED + CACHED + OTHER)
    assert len(calls) == 20


def test_latency_tokens_and_cost_are_summarised():
    summary = usage_report.report(usage_report.chat_calls(ANSWERED + CACHED + [_line(error="boom", latency_ms=5)]))
    assert summary["requests"] == 21 and summary["errors"] == 1
    assert summary["cache_hit_rate"] == 0.5
    assert summary["latency_ms"]["answered"] == {"n": 10, "p50": 1450.0, "p95": 1855.0, "max": 1900.0}
    assert summary["latency_ms"]["cached"]["p95"] == 20.0
    assert summary["steps_ms"]["router"]["p50"] == 300.0
    assert summary["tokens_per_answered"] == 2000
    # Half the questions are cached and cost nothing.
    assert summary["cost_usd_per_1000_questions"] == pytest.approx(0.2)


def test_the_script_writes_its_summary(tmp_path, capsys):
    log = tmp_path / "render.log"
    log.write_text("\n".join(ANSWERED + OTHER), encoding="utf-8")
    out = tmp_path / "usage.json"
    assert usage_report.main([str(log), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["latency_ms"]["answered"]["n"] == 10
    assert "per 1,000 questions" in capsys.readouterr().out
    assert usage_report.main([str(tmp_path / "usage.json")]) == 1  # no chat calls in it


def test_an_eval_runs_cases_count_as_answered_questions(tmp_path):
    run = {
        "results": [
            {"id": f"c{i}", "agent_latency_ms": 2000 + i, "agent_cost_usd": 0.0005,
             "agent_steps_ms": {"route": 900, "generate": 1000 + i, "judge": 5000}}
            for i in range(4)
        ] + [{"id": "old", "agent_cost_usd": 0.1}],  # recorded before latency was
        "errors": [{"id": "e", "error": "429"}],
    }
    calls = usage_report.eval_calls(run)
    summary = usage_report.report(calls)
    assert summary["requests"] == 5 and summary["errors"] == 1 and summary["cache_hit_rate"] == 0
    assert summary["latency_ms"]["answered"]["n"] == 4
    assert set(summary["steps_ms"]) == {"route", "generate"}  # the judge isn't the agent's time
    assert summary["cost_usd_per_1000_questions"] == pytest.approx(0.5)
    path = tmp_path / "eval_x.json"
    path.write_text(json.dumps(run))
    assert usage_report.main([str(path)]) == 0
