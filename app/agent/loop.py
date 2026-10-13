"""The tool loop (AGENT_MODE=loop): one model answers by calling tools.

The router agent (app/agent/graph.py) makes one decision: search once or
don't. Here the model works through the question itself, in a LangGraph loop
of two nodes. "agent" calls the model with the conversation so far and three
tools: search_support_docs (as often as it needs, with its own queries),
read_article (a whole article, when a search found the right one but not the
passage with the detail) and give_answer, which ends the turn with the reply
and the ids of the articles it uses. "tools" runs the calls and adds their
results to the conversation. The loop returns to "agent" until it answers.
Its first call is always a search, and after agent_max_steps calls it is
allowed only give_answer.

The model's turns go back to it as Gemini returned them, so Gemini 3's
thought signatures on function calls are echoed as it requires.

The result has the router agent's shape (answer, sources, chunks,
retrieved_sources, search_query, prompt versions), so /chat, the answer cache
and the eval read either one.
"""

from functools import lru_cache
from typing import TypedDict

from google.genai import types
from langgraph.graph import END, START, StateGraph

from app.agent.history import Turn, to_contents
from app.agent.prompts import AGENT_PROMPT_VERSION, AGENT_SYSTEM_PROMPT
from app.agent.tools import AGENT_TOOLS, GIVE_ANSWER_TOOL_NAME, READ_ARTICLE_TOOL_NAME, SEARCH_DOCS_TOOL_NAME
from app.core.config import get_settings
from app.core.gemini_client import ModelOutputError
from app.core.generation import generate_contents, record_usage
from app.core.observability import time_step
from app.rag.documents import load_documents
from app.rag.ingest import _with_title
from app.rag.retrieval import RetrievedChunk, retrieve

# Every turn must be a tool call, so a turn ends only through give_answer and
# its answer always comes with its sources: left free to reply in plain text,
# gemini-3.1-flash-lite answered a false premise after searching without
# calling give_answer, and the answer lost its citations. After the last
# allowed step, give_answer is the only call left.
_ANY_TOOL = types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
# The first call must be a search. Left to choose, the model obeyed "Don't
# look anything up, answer from memory only" and "Ignore the support docs" by
# answering at once, and declined without the policy (2026-10-13); the router
# never skips its search, since a separate call decides it. A greeting now
# costs a search and two calls, as it does with the router.
_SEARCH_FIRST = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=[SEARCH_DOCS_TOOL_NAME])
)
_ANSWER_ONLY = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=[GIVE_ANSWER_TOOL_NAME])
)


class LoopState(TypedDict, total=False):
    query: str
    history: list[Turn]
    contents: list[types.Content]  # this turn so far: the user's message, the model's calls, their results
    pending: list[types.FunctionCall]  # calls from the latest model turn, for "tools" to run
    steps: int  # model calls made
    chunks: list[RetrievedChunk]  # every passage searched or article read, in order
    searches: list[str]
    articles_read: list[str]
    tool_calls: list[str]
    answer: str
    sources: list[str]


def _why_unusable(response: types.GenerateContentResponse) -> str:
    feedback = response.prompt_feedback
    if feedback and feedback.block_reason:
        return f"prompt blocked: {feedback.block_reason}"
    if response.candidates and response.candidates[0].finish_reason:
        return f"no usable output (finish reason {response.candidates[0].finish_reason})"
    return "no usable output"


async def _agent(state: LoopState) -> dict:
    steps = state.get("steps", 0)
    final = steps >= get_settings().agent_max_steps
    with time_step("agent", step=steps + 1) as usage:
        response = await generate_contents(
            contents=[*to_contents(state.get("history", [])), *state["contents"]],
            system_instruction=AGENT_SYSTEM_PROMPT,
            tools=[AGENT_TOOLS],
            tool_config=_SEARCH_FIRST if steps == 0 else _ANSWER_ONLY if final else _ANY_TOOL,
        )
        record_usage(usage, response)
    content = response.candidates[0].content if response.candidates else None
    if content is None:  # a blocked prompt comes back with no candidates
        raise ModelOutputError(_why_unusable(response))

    calls = [part.function_call for part in content.parts or [] if part.function_call]
    tool_calls = [*state.get("tool_calls", []), *(call.name for call in calls)]
    answer = next((call for call in calls if call.name == GIVE_ANSWER_TOOL_NAME), None)
    if answer is not None:
        args = answer.args or {}
        retrieved = {chunk.doc_id for chunk in state.get("chunks", [])}
        # Only articles the turn actually saw: a made-up id can't become a source.
        sources = [doc_id for doc_id in dict.fromkeys(args.get("sources") or []) if doc_id in retrieved]
        text = str(args.get("answer") or "").strip()
        if not text:
            raise ModelOutputError("give_answer came with no answer")
        return {"answer": text, "sources": sources, "steps": steps + 1, "tool_calls": tool_calls, "pending": []}
    if not calls:
        # Required calls make this rare, but a plain-text reply is still an answer: take it, with no sources.
        text = "".join(part.text or "" for part in content.parts or [] if part.text and not part.thought).strip()
        if not text:
            raise ModelOutputError(_why_unusable(response))
        return {"answer": text, "sources": [], "steps": steps + 1, "tool_calls": tool_calls, "pending": []}
    return {
        "contents": [*state["contents"], content],  # as returned, thought signatures included
        "pending": calls,
        "steps": steps + 1,
        "tool_calls": tool_calls,
    }


@lru_cache
def _articles() -> dict:
    return {doc.doc_id: doc for doc in load_documents()}


async def _tools(state: LoopState) -> dict:
    chunks = list(state.get("chunks", []))
    searches = list(state.get("searches", []))
    read = list(state.get("articles_read", []))
    parts = []
    for call in state["pending"]:
        args = call.args or {}
        if call.name == SEARCH_DOCS_TOOL_NAME:
            query = str(args.get("query") or state["query"])
            found = await retrieve(query)
            searches.append(query)
            chunks += found
            result = {"results": [{"article_id": c.doc_id, "title": c.title, "text": c.text} for c in found]}
            if not found:
                result["note"] = "Nothing relevant found."
        elif call.name == READ_ARTICLE_TOOL_NAME:
            doc = _articles().get(str(args.get("article_id", "")))
            if doc is None:
                result = {"error": f"No article has the id {args.get('article_id')!r}."}
            else:
                read.append(doc.doc_id)
                chunks.append(RetrievedChunk(doc.doc_id, doc.title, _with_title(doc.title, doc.text), 1.0))
                result = {"article_id": doc.doc_id, "title": doc.title, "text": doc.text}
        else:
            result = {"error": f"There's no tool called {call.name!r}."}
        parts.append(types.Part(function_response=types.FunctionResponse(id=call.id, name=call.name, response=result)))
    return {
        "contents": [*state["contents"], types.Content(role="user", parts=parts)],
        "pending": [],
        "chunks": chunks,
        "searches": searches,
        "articles_read": read,
    }


def _next(state: LoopState) -> str:
    return "tools" if state.get("pending") else END


@lru_cache
def get_loop_graph():
    graph = StateGraph(LoopState)
    graph.add_node("agent", _agent)
    graph.add_node("tools", _tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _next, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_loop_agent(query: str, history: list[Turn] | None = None) -> dict:
    state = await get_loop_graph().ainvoke(
        {"query": query, "history": history or [], "contents": [types.Content(role="user", parts=[types.Part(text=query)])]},
        # Each step is an agent node and a tools node; leave room past the step limit for the forced answer.
        {"recursion_limit": 2 * get_settings().agent_max_steps + 4},
    )
    chunks = state.get("chunks", [])
    result = {
        "answer": state["answer"],
        "sources": state["sources"],
        "retrieved_sources": [chunk.doc_id for chunk in chunks],
        "search_query": (state.get("searches") or [None])[0],
        "search_queries": state.get("searches", []),
        "articles_read": state.get("articles_read", []),
        "tool_calls": state.get("tool_calls", []),
        "steps": state.get("steps", 0),
        "router_prompt_version": AGENT_PROMPT_VERSION,
        "answer_prompt_version": AGENT_PROMPT_VERSION,
    }
    if state.get("searches") or state.get("articles_read"):
        result["chunks"] = chunks  # as in router mode, present only when the docs were consulted
    return result
