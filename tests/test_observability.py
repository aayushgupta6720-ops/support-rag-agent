import pytest

from app.core import observability
from app.core.observability import CallTrace, get_trace, start_trace, time_step
from app.core.pricing import embedding_cost_usd, generation_cost_usd


def test_time_step_without_an_active_trace_is_a_no_op():
    observability._current_trace.set(None)
    with time_step("orphan") as usage:
        usage["input_tokens"] = 5
    assert get_trace() is None


def test_time_step_records_usage_and_meta_into_the_active_trace():
    trace = start_trace()
    with time_step("generate", model="m") as usage:
        usage.update(input_tokens=10, output_tokens=4, cost_usd=0.5)

    [step] = trace.steps
    assert (step.name, step.input_tokens, step.output_tokens, step.cost_usd) == ("generate", 10, 4, 0.5)
    assert step.meta == {"model": "m"}
    assert step.latency_ms >= 0


def test_time_step_still_records_when_the_block_raises():
    trace = start_trace()
    with pytest.raises(ValueError):
        with time_step("boom"):
            raise ValueError
    assert [s.name for s in trace.steps] == ["boom"]


def test_trace_totals():
    trace = CallTrace()
    trace.add(observability.StepRecord("a", latency_ms=1.005, input_tokens=3, output_tokens=2, cost_usd=0.1))
    trace.add(observability.StepRecord("b", latency_ms=2.0, input_tokens=1, cost_usd=0.2))
    assert trace.total_tokens == 6
    assert trace.total_cost_usd == pytest.approx(0.3)
    assert trace.total_latency_ms == pytest.approx(3.0, abs=0.01)


def test_pricing_for_known_and_unknown_models():
    assert generation_cost_usd("gemini-flash-lite-latest", 1_000_000, 1_000_000) == pytest.approx(0.375)
    assert embedding_cost_usd("gemini-embedding-001", 1_000_000) == pytest.approx(0.15)
    # unknown models report zero instead of crashing the request
    assert generation_cost_usd("some-new-model", 100, 100) == 0.0
    assert embedding_cost_usd("some-new-model", 100) == 0.0
