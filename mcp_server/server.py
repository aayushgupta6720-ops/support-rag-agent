"""MCP server exposing the support-rag-agent's /chat endpoint as a tool.

This is a thin proxy, not an in-process import of the FastAPI app: `mcp`
pulls in a starlette version newer than what this project's pinned FastAPI
supports, so it lives in its own venv and just calls the running API over
HTTP. Run the main app first (see README), then:

    source mcp_server/venv/bin/activate
    python mcp_server/server.py
"""

import json
import os
import urllib.error
import urllib.request

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SUPPORT_AGENT_URL = os.environ.get("SUPPORT_AGENT_URL", "http://localhost:8000")

mcp = MCPServer(
    name="support-rag-agent",
    description="Answers product support questions using the support-rag-agent RAG service.",
)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The API's own explanation of a failed call (e.g. the daily-quota 503),
    falling back to the status code when there isn't a readable one."""
    try:
        detail = json.loads(exc.read()).get("detail")
    except (ValueError, AttributeError):
        detail = None
    return detail if isinstance(detail, str) else f"The support agent returned HTTP {exc.code}."


@mcp.tool()
def ask_support_agent(query: str, session_id: str | None = None) -> str:
    """Ask the support agent a question and get a grounded answer with sources.
    To ask a follow-up in the same conversation, pass the session_id that
    came with the previous answer."""
    payload = json.dumps({"query": query, "session_id": session_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{SUPPORT_AGENT_URL}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Sized from the agent's worst case, not its typical ~5s: a /chat call
    # makes up to three Gemini calls (route, embed, generate), each allowed
    # gemini_timeout_s (60s), and retries 429s and 503s only in its first
    # 45s (_RETRY_WINDOW_S in app/api/routes.py): 225s at most. A shorter
    # limit here gives up on answers the agent is still about to deliver.
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Only a ToolError's message reaches the model: any other exception
        # becomes a bare "Error executing tool". Pass the API's explanation on
        # so the model can tell the user why, and when to try again.
        raise ToolError(_error_detail(exc)) from exc
    except (TimeoutError, urllib.error.URLError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ToolError(f"Couldn't get an answer from the support agent at {SUPPORT_AGENT_URL} ({reason}).") from exc

    reply = body["answer"]
    sources = body.get("sources", [])
    if sources:
        reply += f"\n\nSources: {', '.join(sources)}"
    if body.get("session_id"):
        reply += f"\n\nsession_id: {body['session_id']}"
    return reply


if __name__ == "__main__":
    mcp.run()
