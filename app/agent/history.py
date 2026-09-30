from dataclasses import dataclass

from google.genai import types


@dataclass(frozen=True)
class Turn:
    """One message of an earlier exchange in the conversation."""

    role: str  # "user" or "model"
    text: str


def to_contents(history: list[Turn]) -> list[types.Content]:
    """Earlier turns as Gemini contents, to go before the latest message. Text
    only: the model's tool calls aren't kept, so there are no function-call
    parts whose thought signatures Gemini 3 would need echoed back."""
    return [types.Content(role=turn.role, parts=[types.Part(text=turn.text)]) for turn in history]
