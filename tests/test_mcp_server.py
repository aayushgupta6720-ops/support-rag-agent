"""The MCP proxy (mcp_server/server.py). It runs in its own venv, because the
mcp package needs a newer starlette than this app's FastAPI; here the mcp
imports are stubbed, so the proxy's own code is what's tested, against a real
local HTTP server standing in for the agent."""

import importlib
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class ToolError(Exception):
    pass


@pytest.fixture(scope="module")
def server_module():
    class MCPServer:
        def __init__(self, **kwargs):
            pass

        def tool(self):
            return lambda fn: fn

    stubs = {
        "mcp": types.ModuleType("mcp"),
        "mcp.server": types.ModuleType("mcp.server"),
        "mcp.server.mcpserver": types.SimpleNamespace(MCPServer=MCPServer),
        "mcp.server.mcpserver.exceptions": types.SimpleNamespace(ToolError=ToolError),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield importlib.import_module("mcp_server.server")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("mcp_server.server", None)


@pytest.fixture
def agent(server_module, monkeypatch):
    """A stand-in /chat: replies with whatever `agent.reply` holds, and keeps what it was sent."""
    state = {"reply": (200, {}), "sent": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["sent"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            status, body = state["reply"]
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(server_module, "SUPPORT_AGENT_URL", f"http://127.0.0.1:{httpd.server_port}")
    yield state
    httpd.shutdown()


def test_an_answer_comes_back_with_its_sources_and_session(server_module, agent):
    agent["reply"] = (200, {"answer": "Refunds take 5-10 days.", "sources": ["billing-refunds"], "session_id": "s1"})

    reply = server_module.ask_support_agent("How long do refunds take?", session_id="s0")

    assert reply == "Refunds take 5-10 days.\n\nSources: billing-refunds\n\nsession_id: s1"
    assert agent["sent"] == [{"query": "How long do refunds take?", "session_id": "s0"}]


def test_the_apis_own_reason_reaches_the_model(server_module, agent):
    agent["reply"] = (429, {"detail": "You've reached the limit of 6 questions a minute. Try again in 60 seconds."})

    with pytest.raises(ToolError, match="limit of 6 questions a minute"):
        server_module.ask_support_agent("hi")


def test_an_unreadable_error_still_says_what_happened(server_module, agent):
    agent["reply"] = (502, ["not", "a", "detail"])

    with pytest.raises(ToolError, match="HTTP 502"):
        server_module.ask_support_agent("hi")


def test_an_unreachable_agent_is_named(server_module, monkeypatch):
    monkeypatch.setattr(server_module, "SUPPORT_AGENT_URL", "http://127.0.0.1:9")

    with pytest.raises(ToolError, match="Couldn't get an answer from the support agent at http://127.0.0.1:9"):
        server_module.ask_support_agent("hi")
