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
    # Drop chunks inside the top k that are unlikely to be relevant, so fewer
    # off-topic docs reach the answer (0 turns either off). min_score is a
    # fixed cosine-similarity floor; max_score_gap drops chunks scoring more
    # than this below the query's best chunk. Scores run higher for some
    # queries than others, so the gap separates better: see
    # scripts/threshold_sweep.py and the README.
    retrieval_min_score: float = 0.0
    retrieval_max_score_gap: float = 0.0
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

    # Shared store (Redis, e.g. a Render Key Value URL) for the rate-limit
    # counts and chat history. Unset keeps both in the process: fine locally,
    # but they're lost on every restart, which on Render's free plan means
    # every time the service wakes up after sleeping.
    redis_url: str = ""
    # Log visitors' questions (and the router's rewrite of them) in full.
    # Off by default: a support chat collects personal details, so the log
    # line records the question's length instead.
    log_chat_text: bool = False
    # Multi-turn chat: how many earlier question-and-answer exchanges a
    # session keeps, and how long an idle session lasts.
    session_max_exchanges: int = 3
    session_ttl_s: int = 1800
    # How long the answer to a conversation's first question is reused for an
    # identical question (see app/api/answer_cache.py). 0 turns caching off.
    answer_cache_ttl_s: int = 3600

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
