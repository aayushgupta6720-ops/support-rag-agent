"""The help-center page and chat widget served at /."""

from fastapi.testclient import TestClient

from app.main import app


def test_the_help_center_page_carries_the_chat_widget():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    for widget_part in ('id="chat-launcher"', 'id="chat-panel"', 'id="chat-log"', 'id="chat-input"'):
        assert widget_part in response.text
    assert "Tidewell is a fictional company" in response.text


def test_the_page_may_only_load_and_call_its_own_origin():
    # answers are rendered as text, and this is the second line of defence
    headers = TestClient(app).get("/").headers

    policy = headers["content-security-policy"]
    for directive in ("default-src 'self'", "script-src 'self'", "connect-src 'self'", "frame-ancestors 'none'"):
        assert directive in policy
    assert headers["x-content-type-options"] == "nosniff"


def test_the_widget_assets_are_served_and_referenced():
    client = TestClient(app)
    page = client.get("/").text

    for path, kind in [("/static/chat.js", "javascript"), ("/static/styles.css", "text/css"),
                       ("/static/favicon.svg", "image/svg+xml")]:
        assert path in page
        response = client.get(path)
        assert response.status_code == 200 and kind in response.headers["content-type"], path


def test_every_doc_has_a_title_for_its_source_chip():
    # sources come back as doc ids; the widget shows each one's title
    from scripts.ingest import load_documents

    widget = TestClient(app).get("/static/chat.js").text
    for doc in load_documents():
        assert f'"{doc.doc_id}":' in widget, doc.doc_id


def test_health_is_still_json_for_the_platform_check():
    assert TestClient(app).get("/health").json()["status"] == "healthy"
