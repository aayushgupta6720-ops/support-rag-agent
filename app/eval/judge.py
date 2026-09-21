from pydantic import BaseModel

from app.core.generation import generate
from app.eval.prompts import JUDGE_SYSTEM_PROMPT


class JudgeVerdict(BaseModel):
    correct: bool
    reasoning: str


async def judge_answer(query: str, reference_answer: str, actual_answer: str) -> JudgeVerdict:
    prompt = (
        f"Question: {query}\n\n"
        f"Reference: {reference_answer}\n\n"
        f"Agent answer: {actual_answer}"
    )
    response = await generate(
        prompt=prompt,
        system_instruction=JUDGE_SYSTEM_PROMPT,
        response_schema=JudgeVerdict,
    )
    return response.parsed
