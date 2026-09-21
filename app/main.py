from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.observability import configure_logging

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


@app.get("/")
async def root() -> dict:
    return {"service": settings.app_name, "status": "ok"}
