# Rough per-token/per-character cost estimates for relative cost tracking
# across calls. These are approximate — check current Gemini API pricing
# before relying on them for real budgeting or billing reconciliation.

GENERATION_USD_PER_1M_TOKENS = {
    "gemini-flash-lite-latest": {"input": 0.075, "output": 0.30},
}

EMBEDDING_USD_PER_1M_CHARS = {
    "gemini-embedding-001": 0.15,
}


def generation_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = GENERATION_USD_PER_1M_TOKENS.get(model)
    if not pricing:
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def embedding_cost_usd(model: str, billable_chars: int) -> float:
    price = EMBEDDING_USD_PER_1M_CHARS.get(model)
    if not price:
        return 0.0
    return billable_chars * price / 1_000_000
