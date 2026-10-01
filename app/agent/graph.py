from functools import lru_cache
from typing import TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agent.history import Turn, to_contents
from app.agent.prompts import (
    DIRECT_ANSWER_PROMPT_VERSION,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    GROUNDED_ANSWER_PROMPT_VERSION,
    GROUNDED_ANSWER_SYSTEM_PROMPT,
    ROUTER_PROMPT_VERSION,
    ROUTER_SYSTEM_PROMPT,
)
from app.agent.tools import SEARCH_DOCS_TOOL
from app.core.config import get_settings
from app.core.gemini_client import ModelOutputError
from app.core.generation import generate
from app.core.observability import time_step
from app.core.pricing import generation_cost_usd
from app.rag.retrieval import RetrievedChunk, retrieve


class AgentAnswer(BaseModel):
    answer: str


class AgentState(TypedDict, total=False):
    query: str
    history: list[Turn]  # earlier turns of the conversation, oldest first
    search_query: str  # what was searched: the router's rewrite of the query
    router_content: types.Content
    chunks: list[RetrievedChunk]
    answer: str
    sources: list[str]
    router_prompt_version: str
    answer_prompt_version: str


def _record_generation_usage(usage: dict, response: types.GenerateContentResponse) -> None:
    metadata = response.usage_metadata
    if metadata is None:
        return
    usage["input_tokens"] = metadata.prompt_token_count or 0
    usage["output_tokens"] = (metadata.candidates_token_count or 0) + (
        metadata.thoughts_token_count or 0
    )
    usage["cost_usd"] = generation_cost_usd(
        get_settings().generation_model, usage["input_tokens"], usage["output_tokens"]
    )


def _why_unusable(response: types.GenerateContentResponse) -> str:
    """Why a response carried no usable output, for ModelOutputError."""
    feedback = response.prompt_feedback
    if feedback and feedback.block_reason:
        return f"prompt blocked: {feedback.block_reason}"
    if response.candidates and response.candidates[0].finish_reason:
        return f"no usable output (finish reason {response.candidates[0].finish_reason})"
    return "no usable output"


async def _route(state: AgentState) -> dict:
    with time_step("route") as usage:
        response = await generate(
            prompt=state["query"],
            system_instruction=ROUTER_SYSTEM_PROMPT,
            tools=[SEARCH_DOCS_TOOL],
            history=to_contents(state.get("history", [])),
        )
        _record_generation_usage(usage, response)
    # A blocked prompt comes back with no candidates rather than an error.
    if not response.candidates or response.candidates[0].content is None:
        raise ModelOutputError(_why_unusable(response))
    return {
        "router_content": response.candidates[0].content,
        "router_prompt_version": ROUTER_PROMPT_VERSION,
    }


def _route_decision(state: AgentState) -> str:
    parts = state["router_content"].parts or []
    has_tool_call = any(part.function_call for part in parts)
    return "retrieve" if has_tool_call else "generate"


async def _retrieve_node(state: AgentState) -> dict:
    parts = state["router_content"].parts or []
    call = next(part.function_call for part in parts if part.function_call)
    # args is None, not {}, when the call came without any
    search_query = (call.args or {}).get("query") or state["query"]
    chunks = await retrieve(search_query)
    return {"chunks": chunks, "search_query": search_query}


async def _generate(state: AgentState) -> dict:
    used_tool = "chunks" in state
    chunks = state.get("chunks") or []

    if used_tool:
        system_instruction = GROUNDED_ANSWER_SYSTEM_PROMPT
        prompt_version = GROUNDED_ANSWER_PROMPT_VERSION
        if chunks:
            context = "\n---\n".join(chunk.text for chunk in chunks)
            prompt = f"Context from support docs:\n{context}\n\nQuestion: {state['query']}"
        else:
            prompt = f"No relevant support docs were found.\n\nQuestion: {state['query']}"
    else:
        system_instruction = DIRECT_ANSWER_SYSTEM_PROMPT
        prompt_version = DIRECT_ANSWER_PROMPT_VERSION
        prompt = state["query"]

    with time_step("generate") as usage:
        response = await generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=AgentAnswer,
            history=to_contents(state.get("history", [])),
        )
        _record_generation_usage(usage, response)
    parsed: AgentAnswer | None = response.parsed
    if parsed is None:  # blocked, or JSON that didn't match AgentAnswer
        raise ModelOutputError(_why_unusable(response))
    return {
        "answer": parsed.answer,
        "sources": [chunk.doc_id for chunk in chunks],
        "answer_prompt_version": prompt_version,
    }


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("route", _route)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("generate", _generate)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route", _route_decision, {"retrieve": "retrieve", "generate": "generate"}
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


@lru_cache
def get_agent_graph():
    return _build_graph()


async def run_agent(query: str, history: list[Turn] | None = None) -> AgentState:
    return await get_agent_graph().ainvoke({"query": query, "history": history or []})


async def route_and_retrieve(query: str, history: list[Turn] | None = None) -> AgentState:
    """The agent's first two steps, without generating an answer: what the
    search saw and found. One model call instead of two, for measuring
    retrieval on its own (scripts/threshold_sweep.py)."""
    state: AgentState = {"query": query, "history": history or []}
    state.update(await _route(state))
    if _route_decision(state) == "retrieve":
        state.update(await _retrieve_node(state))
    return state
