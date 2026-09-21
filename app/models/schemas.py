from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    timestamp: float


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question for the agent.")
    session_id: str | None = Field(
        default=None, description="Optional session/thread id for multi-turn context."
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list,
        description="Ticket/doc IDs the answer was grounded in.",
    )
    latency_ms: float
