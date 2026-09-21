from functools import lru_cache
from typing import TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

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
from app.core.generation import generate
from app.core.observability import time_step
from app.core.pricing import generation_cost_usd
from app.rag.retrieval import RetrievedChunk, retrieve


class AgentAnswer(BaseModel):
    answer: str


class AgentState(TypedDict, total=False):
    query: str
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


async def _route(state: AgentState) -> dict:
    with time_step("route") as usage:
        response = await generate(
            prompt=state["query"],
            system_instruction=ROUTER_SYSTEM_PROMPT,
            tools=[SEARCH_DOCS_TOOL],
        )
        _record_generation_usage(usage, response)
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
    search_query = call.args.get("query") or state["query"]
    chunks = await retrieve(search_query)
    return {"chunks": chunks}


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
        )
        _record_generation_usage(usage, response)
    parsed: AgentAnswer = response.parsed
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


async def run_agent(query: str) -> AgentState:
    return await get_agent_graph().ainvoke({"query": query})
