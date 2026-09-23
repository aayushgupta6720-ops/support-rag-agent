"""Scripted stand-ins for Gemini responses, so agent tests run offline."""

from google.genai import types
from pydantic import BaseModel

from app.rag.retrieval import RetrievedChunk


def model_response(
    *,
    function_call: tuple[str, dict] | None = None,
    parsed: BaseModel | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> types.GenerateContentResponse:
    parts = []
    if function_call:
        name, args = function_call
        parts.append(types.Part(function_call=types.FunctionCall(name=name, args=args)))
    else:
        parts.append(types.Part(text="ok"))
    response = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=input_tokens, candidates_token_count=output_tokens
        ),
    )
    response.parsed = parsed
    return response


class FakeGenerate:
    """Replaces app.core.generation.generate: returns scripted responses in
    order and records every call's kwargs."""

    def __init__(self, responses: list[types.GenerateContentResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> types.GenerateContentResponse:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def chunk(doc_id: str, text: str = "some text", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, title=doc_id.title(), text=text, score=score)
