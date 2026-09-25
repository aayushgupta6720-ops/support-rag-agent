import pytest

import app.core.generation
import app.main
import app.rag.embeddings
import app.rag.qdrant_store
from app.api.ratelimit import build_rate_limiters
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


@pytest.fixture(autouse=True)
def fresh_rate_limits(monkeypatch):
    """Each test starts with no requests counted, as after a restart."""
    monkeypatch.setattr(app.main.app.state, "rate_limiters", build_rate_limiters(get_settings()))
