from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "support-rag-agent"
    environment: str = "development"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "support_docs"
    gemini_api_key: str = ""
    # Per-request limit on Gemini calls. A slow-but-working call has taken up
    # to ~47s during a demand spike; past this, a stalled call is abandoned.
    gemini_timeout_s: float = 60.0
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    generation_model: str = "gemini-flash-lite-latest"
    retrieval_top_k: int = 4
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 100

    # Per-visitor limits on /chat (see app/api/ratelimit.py), so one visitor
    # can't use up the shared daily Gemini quota. 0 turns a limit off.
    chat_limit_per_minute: int = 6
    chat_limit_per_day: int = 30
    # A header holding the visitor's IP, set by a proxy in front of the app
    # that overwrites any client-sent copy (behind Cloudflare, as on Render:
    # CF-Connecting-IP). Unset means the connecting address, right when
    # nothing sits in front. Never X-Forwarded-For: visitors can forge it.
    client_ip_header: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
