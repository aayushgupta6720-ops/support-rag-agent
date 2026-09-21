import time

from fastapi import APIRouter

from app.agent.graph import run_agent
from app.core.observability import log_event, start_trace
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — also doubles as an uptime probe once deployed (Step 6)."""
    return HealthResponse(status="healthy", timestamp=time.time())


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
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
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            steps=trace.as_dicts(),
        )
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
