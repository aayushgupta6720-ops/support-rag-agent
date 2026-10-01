from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.body_limit import BodySizeLimit
from app.api.answer_cache import build_answer_cache
from app.api.feedback import build_feedback_store
from app.api.feedback import router as feedback_router
from app.api.ratelimit import build_rate_limiters
from app.api.routes import router
from app.api.sessions import build_session_store
from app.core.config import get_settings
from app.core.observability import configure_logging, log_event
from app.core.redis_client import get_redis

settings = get_settings()
configure_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Agentic RAG service scaffold — async FastAPI backend for a "
        "support-ticket RAG agent."
    ),
)

app.include_router(router)
app.include_router(feedback_router)
redis = get_redis()
app.state.rate_limiters = build_rate_limiters(settings, redis)
app.state.sessions = build_session_store(settings, redis)
app.state.answer_cache = build_answer_cache(settings, redis)
app.state.feedback = build_feedback_store(settings, redis)
# Which store the rate limits and chat history use, to check after a deploy.
log_event(event="state_store", backend="redis" if redis else "memory")
# A /chat body is at most a few KB: 2,000 chars of query even fully escaped.
app.add_middleware(BodySizeLimit, max_bytes=64 * 1024)


# The help-center page and its chat widget (app/static/). Everything it
# loads comes from this origin, so the policy allows nothing else: even if
# markup slipped into an answer, it couldn't load or run anything.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")
_PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # so a deploy's new ?v= asset URLs are picked up straight away
    "Cache-Control": "no-cache",
}


@app.get("/", include_in_schema=False)
async def help_center() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html", headers=_PAGE_HEADERS)
