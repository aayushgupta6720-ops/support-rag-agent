JUDGE_PROMPT_VERSION_V1 = "judge_v1"
JUDGE_SYSTEM_PROMPT_V1 = (
    "You are grading a support agent's answer against a reference. The "
    "reference either states the facts a correct answer must contain, or "
    "describes what a correct response should do (e.g. decline to answer, "
    "greet the user). Mark `correct: true` only if the agent's answer "
    "satisfies the reference — matching its key facts or behavior — and "
    "does not contradict it or assert unsupported claims. Minor wording "
    "differences don't matter. Briefly explain your reasoning."
)

# v1 failed a correct answer (eval: reasoning_lost_app_have_codes) for a
# true caveat the reference didn't mention: without the docs, it couldn't
# tell a supported extra detail from an invented one. v2 is shown the docs
# the agent retrieved, told that being more complete than the reference is
# fine when those docs back it up, and grades multi-turn cases, where the
# question only makes sense after the earlier conversation.
JUDGE_PROMPT_VERSION = "judge_v2"
JUDGE_SYSTEM_PROMPT_V2 = (
    "You are grading a support agent's answer against a reference. The "
    "reference either states the facts a correct answer must contain, or "
    "describes what a correct response should do (e.g. decline to answer, "
    "greet the user). Mark `correct: true` only if the agent's answer "
    "satisfies the reference — matching its key facts or behavior — and "
    "does not contradict it or assert unsupported claims. Details or "
    "caveats beyond the reference are fine when the support docs the agent "
    "retrieved (shown when there are any) back them up: don't fail an "
    "answer for being more complete than the reference. Minor wording "
    "differences don't matter. If an earlier conversation is shown, the "
    "question is its latest message. Briefly explain your reasoning."
)

JUDGE_SYSTEM_PROMPT = JUDGE_SYSTEM_PROMPT_V2
