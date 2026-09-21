import contextvars
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("support_rag_agent")


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


@dataclass
class StepRecord:
    name: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class CallTrace:
    steps: list[StepRecord] = field(default_factory=list)

    def add(self, record: StepRecord) -> None:
        self.steps.append(record)

    @property
    def total_latency_ms(self) -> float:
        return round(sum(step.latency_ms for step in self.steps), 2)

    @property
    def total_tokens(self) -> int:
        return sum(step.input_tokens + step.output_tokens for step in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(step.cost_usd for step in self.steps), 8)

    def as_dicts(self) -> list[dict]:
        return [asdict(step) for step in self.steps]


_current_trace: contextvars.ContextVar["CallTrace | None"] = contextvars.ContextVar(
    "current_trace", default=None
)


def start_trace() -> CallTrace:
    trace = CallTrace()
    _current_trace.set(trace)
    return trace


def get_trace() -> "CallTrace | None":
    return _current_trace.get()


@contextmanager
def time_step(name: str, **meta):
    """Time a block and, if a trace is active, record it with whatever
    token/cost usage the caller fills into the yielded dict."""
    start = time.perf_counter()
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    try:
        yield usage
    finally:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        trace = get_trace()
        if trace is not None:
            trace.add(
                StepRecord(
                    name=name,
                    latency_ms=latency_ms,
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    cost_usd=usage["cost_usd"],
                    meta=meta,
                )
            )


def log_event(**fields) -> None:
    logger.info(json.dumps(fields, default=str))
