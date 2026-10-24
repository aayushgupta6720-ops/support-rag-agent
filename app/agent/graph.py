from functools import lru_cache
from typing import TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agent.history import Turn, to_contents
from app.agent.prompts import (
    DIRECT_ANSWER_PROMPT_VERSION,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    FULL_CONTEXT_PROMPT_VERSION,
    FULL_CONTEXT_SYSTEM_PROMPT,
    GROUNDED_ANSWER_PROMPT_VERSION,
    GROUNDED_ANSWER_SYSTEM_PROMPT,
    ROUTER_PROMPT_VERSION,
    ROUTER_SYSTEM_PROMPT,
)
from app.agent.tools import SEARCH_DOCS_TOOL
from app.core.config import get_settings
from app.core.gemini_client import ModelOutputError
from app.core.generation import generate, record_usage
from app.core.observability import time_step
from app.rag.documents import load_documents
from app.rag.ingest import _with_title
from app.rag.retrieval import RetrievedChunk, retrieve


class AgentAnswer(BaseModel):
    answer: str


class GroundedAnswer(AgentAnswer):
    """An answer from retrieved docs, and which of them it relies on."""

    sources: list[str] = Field(default_factory=list, description="ids of the docs the answer uses")


class AgentState(TypedDict, total=False):
    query: str
    history: list[Turn]  # earlier turns of the conversation, oldest first
    search_query: str  # what was searched: the router's rewrite of the query
    router_content: types.Content
    chunks: list[RetrievedChunk]
    answer: str
    sources: list[str]  # the docs the answer says it used
    retrieved_sources: list[str]  # one per chunk retrieved, best first
    router_prompt_version: str
    answer_prompt_version: str


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
        record_usage(usage, response)
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
    return await _answer(state, GROUNDED_ANSWER_SYSTEM_PROMPT, GROUNDED_ANSWER_PROMPT_VERSION)


async def _answer(state: AgentState, grounded_prompt: str, grounded_version: str) -> dict:
    """The answer call: from the chunks in `state` with `grounded_prompt` when
    there are any (a search ran), else the direct-answer prompt."""
    used_tool = "chunks" in state
    chunks = state.get("chunks") or []

    if used_tool:
        system_instruction = grounded_prompt
        prompt_version = grounded_version
        schema = GroundedAnswer
        if chunks:
            # labelled, so the answer can say which docs it used
            context = "\n---\n".join(f"[doc: {chunk.doc_id}]\n{chunk.text}" for chunk in chunks)
            prompt = f"Context from support docs:\n{context}\n\nQuestion: {state['query']}"
        else:
            prompt = f"No relevant support docs were found.\n\nQuestion: {state['query']}"
    else:
        system_instruction = DIRECT_ANSWER_SYSTEM_PROMPT
        prompt_version = DIRECT_ANSWER_PROMPT_VERSION
        schema = AgentAnswer
        prompt = state["query"]

    with time_step("generate") as usage:
        response = await generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_schema=schema,
            history=to_contents(state.get("history", [])),
        )
        record_usage(usage, response)
    parsed: AgentAnswer | None = response.parsed
    if parsed is None:  # blocked, or JSON that didn't match AgentAnswer
        raise ModelOutputError(_why_unusable(response))
    retrieved = [chunk.doc_id for chunk in chunks]
    # Only ids that were actually retrieved: a made-up one can't become a source.
    cited = [doc_id for doc_id in dict.fromkeys(getattr(parsed, "sources", None) or []) if doc_id in retrieved]
    return {
        "answer": parsed.answer,
        "sources": cited,  # the docs the answer says it used
        "retrieved_sources": retrieved,  # one per chunk retrieved, best first
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


@lru_cache
def all_articles() -> tuple[RetrievedChunk, ...]:
    """Every help article whole, headed by its title as ingest heads a chunk."""
    return tuple(RetrievedChunk(d.doc_id, d.title, _with_title(d.title, d.text), 1.0) for d in load_documents())


async def run_full_context(query: str, history: list[Turn] | None = None) -> AgentState:
    """AGENT_MODE=full_context: one model call, the grounded answer with every
    article as its context, instead of the router's call, a search and the
    answer's call. The 11 articles are about 8,700 characters, so the whole help
    center fits in a prompt, and the free tier's daily request quota stretches
    twice as far. The router stays for a help center too large to send whole."""
    state: AgentState = {"query": query, "history": history or [], "chunks": list(all_articles())}
    return {**state, **await _answer(state, FULL_CONTEXT_SYSTEM_PROMPT, FULL_CONTEXT_PROMPT_VERSION)}


async def run_agent(query: str, history: list[Turn] | None = None) -> AgentState:
    """Answer a question with the agent AGENT_MODE names: this router agent,
    the tool loop (app/agent/loop.py), or one call with every article
    (run_full_context). All return the same shape."""
    mode = get_settings().agent_mode
    if mode == "loop":
        from app.agent.loop import run_loop_agent

        return await run_loop_agent(query, history)
    if mode == "full_context":
        return await run_full_context(query, history)
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
