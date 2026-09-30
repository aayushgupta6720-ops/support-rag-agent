from dataclasses import dataclass

from pydantic import BaseModel

from app.agent.history import Turn
from app.core.generation import generate
from app.eval.prompts import JUDGE_SYSTEM_PROMPT

# One verdict is noisy: the same answer text has been graded both ways on
# different runs. So a failing verdict is re-judged, up to MAX_VOTES in all,
# and the majority kept, stopping as soon as one side has it. A passing first
# verdict isn't rechecked, to save quota: the flips seen so far were correct
# answers marked wrong, so the extra calls go where the noise was. A wrong
# answer that fools the judge on the first vote still passes, as it did
# before.
MAX_VOTES = 3


class JudgeVerdict(BaseModel):
    """One vote, in the shape the judge model returns."""

    correct: bool
    reasoning: str


@dataclass
class JudgeResult:
    correct: bool
    reasoning: str  # from the first vote on the winning side
    votes: list[bool]


def _judge_prompt(query: str, reference_answer: str, actual_answer: str, history: list[Turn], docs: list[str]) -> str:
    parts = []
    if history:
        lines = "\n".join(f"{'User' if turn.role == 'user' else 'Agent'}: {turn.text}" for turn in history)
        parts.append(f"Earlier conversation:\n{lines}")
    if docs:
        parts.append("Support docs the agent retrieved:\n" + "\n---\n".join(docs))
    parts += [f"Question: {query}", f"Reference: {reference_answer}", f"Agent answer: {actual_answer}"]
    return "\n\n".join(parts)


async def judge_answer(
    query: str,
    reference_answer: str,
    actual_answer: str,
    history: list[Turn] | None = None,
    docs: list[str] | None = None,
) -> JudgeResult:
    prompt = _judge_prompt(query, reference_answer, actual_answer, history or [], docs or [])
    majority = MAX_VOTES // 2 + 1
    votes: list[JudgeVerdict] = []
    while True:
        response = await generate(prompt=prompt, system_instruction=JUDGE_SYSTEM_PROMPT, response_schema=JudgeVerdict)
        votes.append(response.parsed)
        passes = sum(vote.correct for vote in votes)
        if votes[0].correct or passes >= majority or len(votes) - passes >= majority:
            break
    correct = passes > len(votes) - passes
    reasoning = next(vote.reasoning for vote in votes if vote.correct == correct)
    return JudgeResult(correct=correct, reasoning=reasoning, votes=[vote.correct for vote in votes])
