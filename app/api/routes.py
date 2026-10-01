import math
import secrets
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agent.graph import run_agent
from app.api.answer_cache import answer_cache_key
from app.api.ratelimit import client_key, rate_limit
from app.core.config import get_settings
from app.core.gemini_client import (
    DailyQuotaExhaustedError,
    ModelOutputError,
    ModelOverloadedError,
    ModelTimeoutError,
    RateLimitedError,
    next_daily_quota_reset,
    start_retry_window,
)
from app.core.observability import log_event, start_trace
from app.models.schemas import ArticleResponse, ChatRequest, ChatResponse, HealthResponse
from app.rag.documents import load_documents
from app.rag.ingest import Document
from app.rag.qdrant_store import SearchUnavailableError

router = APIRouter()

# How long into a /chat its Gemini calls may still retry 429s and 503s.
# After it, the next failure is returned as it comes, so a /chat ends within
# this plus its three calls' gemini_timeout_s: 45 + 3 x 60 = 225s, inside
# the MCP proxy's 240s with room for Qdrant. Change both together.
_RETRY_WINDOW_S = 45.0


def _in_about(seconds: float) -> str:
    if seconds >= 3600:
        hours = round(seconds / 3600)
        return f"in about {hours} hour{'s' if hours != 1 else ''}"
    minutes = max(1, round(seconds / 60))
    return f"in about {minutes} minute{'s' if minutes != 1 else ''}"


_HANDLED_ERRORS = (
    DailyQuotaExhaustedError,
    RateLimitedError,
    ModelOverloadedError,
    ModelTimeoutError,
    ModelOutputError,
    SearchUnavailableError,
)


def _error_response(exc: Exception) -> JSONResponse:
    """What /chat returns instead of a bare 500 when an answer can't be made:
    Gemini's quota used up, the model overloaded, slow, or returning nothing
    usable, or the search down. `detail` is written for a person (the chat
    widget shows it as is); Retry-After (and resets_at, for the daily quota)
    are for clients that want to schedule a retry."""
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
    if isinstance(exc, ModelTimeoutError):
        detail = (
            f"Gemini didn't respond within {get_settings().gemini_timeout_s:g} seconds, so the "
            "request was stopped. It's usually a temporary slowdown on Google's side; "
            "try again in a minute."
        )
        return JSONResponse(status_code=504, content={"detail": detail}, headers={"Retry-After": "60"})
    if isinstance(exc, ModelOverloadedError):
        detail = (
            "Gemini is overloaded right now (a temporary Google-side issue, not "
            "a problem with your question). Try again in a minute."
        )
        return JSONResponse(status_code=503, content={"detail": detail}, headers={"Retry-After": "60"})
    if isinstance(exc, ModelOutputError):
        detail = "The assistant couldn't produce an answer to that one. Try rephrasing your question."
        return JSONResponse(status_code=502, content={"detail": detail})
    if isinstance(exc, SearchUnavailableError):
        detail = (
            "Searching the help articles failed just now (a temporary problem on our side). "
            "Try again in a minute."
        )
        return JSONResponse(status_code=503, content={"detail": detail}, headers={"Retry-After": "30"})
    detail = (
        "The Gemini API is getting more requests than this demo's quota allows. "
        "Wait a minute and try again."
    )
    return JSONResponse(status_code=429, content={"detail": detail}, headers={"Retry-After": "60"})


def _chat_text(query: str, search_query: str | None = None) -> dict:
    """The question's text for the log line, only if LOG_CHAT_TEXT is on;
    otherwise just its length."""
    if get_settings().log_chat_text:
        return {"query": query, "search_query": search_query}
    return {"query_chars": len(query)}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check — also doubles as an uptime probe once deployed (Step 6)."""
    return HealthResponse(status="healthy", timestamp=time.time())


@router.post("/chat", response_model=ChatResponse, dependencies=[rate_limit("chat")])
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse | JSONResponse:
    """
    Agentic chat endpoint.

    Runs the LangGraph agent: a router decides whether the query needs the
    support-docs retrieval tool or can be answered directly, then generation
    produces a structured, schema-validated answer grounded in whatever
    context was retrieved. The session's earlier exchanges go to both, so
    follow-up questions work. Emits one structured log line per call with
    latency/cost/quality signal for observability.
    """
    start = time.perf_counter()
    trace = start_trace()
    start_retry_window(_RETRY_WINDOW_S)

    # A new conversation gets an unguessable id to send back for follow-ups.
    session_id = request.session_id or secrets.token_urlsafe(16)
    session_key = f"{client_key(http_request)}|{session_id}"  # see app/api/sessions.py
    sessions = http_request.app.state.sessions
    history = await sessions.history(session_key)

    # A conversation's first question may already have an answer (the topic
    # cards send the same few); a follow-up depends on its history.
    cache = http_request.app.state.answer_cache
    cache_key = answer_cache_key(request.query) if cache is not None and not history else None
    result = await cache.get(cache_key) if cache_key else None
    cache_hit = result is not None

    if not cache_hit:
        try:
            result = await run_agent(request.query, history=history)
        except Exception as exc:
            log_event(
                event="chat_call",
                session_id=session_id,
                history_turns=len(history),
                **_chat_text(request.query),
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                steps=trace.as_dicts(),
            )
            if isinstance(exc, _HANDLED_ERRORS):
                return _error_response(exc)
            raise
        if cache_key:
            await cache.put(cache_key, {
                "answer": result["answer"],
                "sources": list(dict.fromkeys(result.get("sources", []))),
                "router_prompt_version": result.get("router_prompt_version"),
                "answer_prompt_version": result.get("answer_prompt_version"),
            })

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    answer = result["answer"]
    sources = result.get("sources", [])
    chunks = result.get("chunks") or []
    await sessions.add_exchange(session_key, request.query, answer)

    log_event(
        event="chat_call",
        session_id=session_id,
        history_turns=len(history),
        cache_hit=cache_hit,
        **_chat_text(request.query, result.get("search_query")),
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
        # One entry per retrieved chunk, so a doc with two chunks in the top k
        # was listed twice. The log line above keeps them per chunk.
        sources=list(dict.fromkeys(sources)),
        latency_ms=elapsed_ms,
        session_id=session_id,
        answer_id=uuid.uuid4().hex,
    )


@lru_cache
def _articles() -> dict[str, Document]:
    return {doc.doc_id: doc for doc in load_documents()}


@router.get("/articles/{doc_id}", response_model=ArticleResponse)
async def article(doc_id: str) -> ArticleResponse:
    """A help article, for the chat widget's source links. Looked up by id
    in the loaded docs, never used as a path."""
    doc = _articles().get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="There's no help article with that id.")
    return ArticleResponse(doc_id=doc.doc_id, title=doc.title, body=doc.text)
