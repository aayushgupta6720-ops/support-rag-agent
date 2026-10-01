import pytest

import app.core.generation
import app.main
import app.rag.embeddings
import app.rag.qdrant_store
from app.api.ratelimit import build_rate_limiters
from app.api.sessions import build_session_store
from app.core.config import get_settings


def _no_network(*args, **kwargs):
    raise AssertionError("tests must not reach Gemini or Qdrant; patch the call instead")


@pytest.fixture(autouse=True)
def block_external_services(monkeypatch):
    """Every test runs offline. Anything that isn't faked fails loudly
    instead of silently spending API quota."""
    monkeypatch.setattr(app.core.generation, "get_gemini_client", _no_network)
    monkeypatch.setattr(app.rag.embeddings, "get_gemini_client", _no_network)
    monkeypatch.setattr(app.rag.qdrant_store, "get_client", _no_network)
    # A REDIS_URL in a local .env must not point tests at a real Redis; tests
    # that need one pass a fakeredis client in.
    monkeypatch.setattr(get_settings(), "redis_url", "")
    # Likewise a RETRIEVAL_MIN_SCORE there must not change what tests retrieve.
    monkeypatch.setattr(get_settings(), "retrieval_min_score", 0.0)
    monkeypatch.setattr(get_settings(), "retrieval_max_score_gap", 0.0)
    monkeypatch.setattr(get_settings(), "log_chat_text", False)


@pytest.fixture(autouse=True)
def fresh_app_state(monkeypatch):
    """Each test starts with no requests counted and no chat history."""
    monkeypatch.setattr(app.main.app.state, "rate_limiters", build_rate_limiters(get_settings()))
    monkeypatch.setattr(app.main.app.state, "sessions", build_session_store(get_settings()))
