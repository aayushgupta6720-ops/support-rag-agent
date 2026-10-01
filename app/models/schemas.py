from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class HealthResponse(BaseModel):
    status: str
    timestamp: float


class ChatRequest(BaseModel):
    # Each query goes to Gemini up to three times and into the logs, so an
    # unbounded one can use up the per-minute token quota for everyone. The
    # longest golden-set question is ~150 chars; this leaves room for a
    # pasted error message. Surrounding whitespace is stripped first, so a
    # whitespace-only query is refused instead of costing two model calls.
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)] = Field(
        ..., description="User's question for the agent."
    )
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
    answer_id: str = Field(description="Identifies this answer when rating it at POST /feedback.")
    history_turns: int = Field(
        description="How many earlier messages of the conversation this answer could see. 0 for a "
        "follow-up means the session had expired (after 30 idle minutes) and it was answered on its own."
    )


class ArticleResponse(BaseModel):
    doc_id: str
    title: str
    body: str = Field(description="The article's Markdown, without its title heading.")


class FeedbackRequest(BaseModel):
    """A visitor's rating of one answer. The question and answer come from
    the visitor's own chat, sent only when they choose to rate it."""

    answer_id: str = Field(..., min_length=1, max_length=64)
    rating: Literal["up", "down"]
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=8000)
    sources: list[Annotated[str, StringConstraints(max_length=64)]] = Field(default_factory=list, max_length=10)
