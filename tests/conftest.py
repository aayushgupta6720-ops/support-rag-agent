import pytest

import app.core.generation
import app.rag.embeddings
import app.rag.qdrant_store


def _no_network(*args, **kwargs):
    raise AssertionError("tests must not reach Gemini or Qdrant; patch the call instead")


@pytest.fixture(autouse=True)
def block_external_services(monkeypatch):
    """Every test runs offline. Anything that isn't faked fails loudly
    instead of silently spending API quota."""
    monkeypatch.setattr(app.core.generation, "get_gemini_client", _no_network)
    monkeypatch.setattr(app.rag.embeddings, "get_gemini_client", _no_network)
    monkeypatch.setattr(app.rag.qdrant_store, "get_client", _no_network)
