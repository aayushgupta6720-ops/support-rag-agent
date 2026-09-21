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
import urllib.request

from mcp.server.mcpserver import MCPServer

SUPPORT_AGENT_URL = os.environ.get("SUPPORT_AGENT_URL", "http://localhost:8000")

mcp = MCPServer(
    name="support-rag-agent",
    description="Answers product support questions using the support-rag-agent RAG service.",
)


@mcp.tool()
def ask_support_agent(query: str, session_id: str | None = None) -> str:
    """Ask the support agent a question and get a grounded answer with sources."""
    payload = json.dumps({"query": query, "session_id": session_id}).encode("utf-8")
    request = urllib.request.Request(
        f"{SUPPORT_AGENT_URL}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())

    answer = body["answer"]
    sources = body.get("sources", [])
    if sources:
        return f"{answer}\n\nSources: {', '.join(sources)}"
    return answer


if __name__ == "__main__":
    mcp.run()
