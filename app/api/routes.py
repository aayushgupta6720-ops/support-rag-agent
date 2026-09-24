import math
import time
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.agent.graph import run_agent
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    ModelOverloadedError,
    RateLimitedError,
    next_daily_quota_reset,
)
from app.core.observability import log_event, start_trace
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


def _in_about(seconds: float) -> str:
    if seconds >= 3600:
        hours = round(seconds / 3600)
        return f"in about {hours} hour{'s' if hours != 1 else ''}"
    minutes = max(1, round(seconds / 60))
    return f"in about {minutes} minute{'s' if minutes != 1 else ''}"


_GEMINI_UNAVAILABLE = (DailyQuotaExhaustedError, RateLimitedError, ModelOverloadedError)


def _gemini_unavailable_response(
    exc: DailyQuotaExhaustedError | RateLimitedError | ModelOverloadedError,
) -> JSONResponse:
    """What /chat returns instead of a bare 500 when Gemini can't serve the
    call: quota used up, or the model overloaded. `detail` is written for a
    person; Retry-After (and resets_at, for the daily quota) are for clients
    that want to schedule a retry."""
    if isinstance(exc, DailyQuotaExhaustedError):
        resets_at = next_daily_quota_reset()
        seconds = max(0.0, (resets_at - datetime.now(timezone.utc)).total_seconds())
        detail = (
            "The Gemini API's daily quota for this demo is used up. It resets at "
            f"midnight Pacific time ({_in_about(seconds)}); please try again after that."
        )
        return JSONResponse(
            status_code=503,
            content={"detail": detail, "resets_at": resets_at.isoformat()},
            headers={"Retry-After": str(math.ceil(seconds))},
        )
    if isinstance(exc, ModelOverloadedError):
        detail = (
            "Gemini is overloaded right now (a temporary Google-side issue, not "
            "a problem with your question). Try again in a minute."
        )
        return JSONResponse(status_code=503, content={"detail": detail}, headers={"Retry-After": "60"})
    detail = (
        "The Gemini API is getting more requests than this demo's quota allows. "
        "Wait a minute and try again."
    )
    return JSONResponse(status_code=429, content={"detail": detail}, headers={"Retry-After": "60"})


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — also doubles as an uptime probe once deployed (Step 6)."""
    return HealthResponse(status="healthy", timestamp=time.time())


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse | JSONResponse:
    """
    Agentic chat endpoint.

    Runs the LangGraph agent: a router decides whether the query needs the
    support-docs retrieval tool or can be answered directly, then generation
    produces a structured, schema-validated answer grounded in whatever
    context was retrieved. Emits one structured log line per call with
    latency/cost/quality signal for observability.
    """
    start = time.perf_counter()
    trace = start_trace()

    try:
        result = await run_agent(request.query)
    except Exception as exc:
        log_event(
            event="chat_call",
            session_id=request.session_id,
            query=request.query,
            error=str(exc),
            error_type=type(exc).__name__,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            steps=trace.as_dicts(),
        )
        if isinstance(exc, _GEMINI_UNAVAILABLE):
            return _gemini_unavailable_response(exc)
        raise

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    answer = result["answer"]
    sources = result.get("sources", [])
    chunks = result.get("chunks") or []

    log_event(
        event="chat_call",
        session_id=request.session_id,
        query=request.query,
        answer_length=len(answer),
        sources=sources,
        used_tool="chunks" in result,
        num_chunks_retrieved=len(chunks),
        retrieval_scores=[round(chunk.score, 4) for chunk in chunks],
        router_prompt_version=result.get("router_prompt_version"),
        answer_prompt_version=result.get("answer_prompt_version"),
        latency_ms=elapsed_ms,
        total_tokens=trace.total_tokens,
        total_cost_usd=trace.total_cost_usd,
        steps=trace.as_dicts(),
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        latency_ms=elapsed_ms,
    )
