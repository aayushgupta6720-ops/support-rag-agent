# Rough per-token cost estimates for relative cost tracking across calls,
# from https://ai.google.dev/gemini-api/docs/pricing (paid tier, Standard),
# checked 2026-10-01. On the free tier the actual bill is $0; these say what
# the same traffic would cost on a paid key. Check current pricing before
# relying on them for budgeting.

GENERATION_USD_PER_1M_TOKENS = {
    # gemini-flash-lite-latest resolves to gemini-3.5-flash-lite
    "gemini-flash-lite-latest": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
}

# gemini-embedding-001 is no longer on the pricing page (Gemini Embedding 2
# is, at $0.20); this is its last published price.
EMBEDDING_USD_PER_1M_TOKENS = {
    "gemini-embedding-001": 0.15,
}

# The embed call reports billable characters, not tokens; English text runs
# about 4 characters per token.
_CHARS_PER_TOKEN = 4


def generation_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = GENERATION_USD_PER_1M_TOKENS.get(model)
    if not pricing:
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def embedding_cost_usd(model: str, billable_chars: int) -> float:
    price = EMBEDDING_USD_PER_1M_TOKENS.get(model)
    if not price:
        return 0.0
    return billable_chars / _CHARS_PER_TOKEN * price / 1_000_000
