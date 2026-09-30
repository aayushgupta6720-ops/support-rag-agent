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
        default=None,
        max_length=64,
        description="The session_id from an earlier response, to ask a follow-up in that conversation. "
        "Omit it to start a new one.",
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(
        default_factory=list,
        description="Ticket/doc IDs the answer was grounded in.",
    )
    latency_ms: float
    session_id: str = Field(description="Send this back with the next question to continue the conversation.")
