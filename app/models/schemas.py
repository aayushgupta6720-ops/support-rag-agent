from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: float


class ChatRequest(BaseModel):
    # Each query goes to Gemini up to three times and into the logs, so an
    # unbounded one can use up the per-minute token quota for everyone. The
    # longest golden-set question is ~150 chars; this leaves room for a
    # pasted error message.
    query: str = Field(..., min_length=1, max_length=2000, description="User's question for the agent.")
    session_id: str | None = Field(
        default=None, max_length=64, description="Optional session/thread id for multi-turn context."
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list,
        description="Ticket/doc IDs the answer was grounded in.",
    )
    latency_ms: float
