JUDGE_PROMPT_VERSION = "judge_v1"
JUDGE_SYSTEM_PROMPT_V1 = (
    "You are grading a support agent's answer against a reference. The "
    "reference either states the facts a correct answer must contain, or "
    "describes what a correct response should do (e.g. decline to answer, "
    "greet the user). Mark `correct: true` only if the agent's answer "
    "satisfies the reference — matching its key facts or behavior — and "
    "does not contradict it or assert unsupported claims. Minor wording "
    "differences don't matter. Briefly explain your reasoning."
)

JUDGE_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT_V1
