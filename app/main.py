from fastapi import FastAPI

from app.api.body_limit import BodySizeLimit
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
redis = get_redis()
app.state.rate_limiters = build_rate_limiters(settings, redis)
app.state.sessions = build_session_store(settings, redis)
# Which store the rate limits and chat history use, to check after a deploy.
log_event(event="state_store", backend="redis" if redis else "memory")
# A /chat body is at most a few KB: 2,000 chars of query even fully escaped.
app.add_middleware(BodySizeLimit, max_bytes=64 * 1024)


@app.get("/")
async def root() -> dict:
    return {"service": settings.app_name, "status": "ok"}
