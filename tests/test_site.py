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


def test_every_help_article_is_served_by_its_id():
    from app.rag.documents import load_documents

    client = TestClient(app)
    for doc in load_documents():
        article = client.get(f"/articles/{doc.doc_id}").json()
        assert (article["doc_id"], article["title"]) == (doc.doc_id, doc.title)
        assert article["body"] == doc.text and not article["body"].startswith("#")


def test_an_unknown_or_path_like_article_id_is_a_404():
    client = TestClient(app)
    for doc_id in ("no-such-article", "..%2F..%2Fapp%2Fmain", "%2E%2E", "README"):
        assert client.get(f"/articles/{doc_id}").status_code == 404, doc_id


def test_the_widget_links_sources_to_articles_and_can_rate_answers():
    widget = TestClient(app).get("/static/chat.js").text
    assert '"/articles/" + encodeURIComponent(docId)' in widget
    assert 'fetch("/feedback"' in widget


def test_a_question_in_the_hero_box_is_kept_while_an_answer_is_coming():
    # send() turns it away while busy ("One moment..."), so emptying the box
    # first lost the question; the chat composer already keeps its text
    widget = TestClient(app).get("/static/chat.js").text
    hero_submit = widget.split('$("hero-ask").addEventListener("submit"', 1)[1].split("});", 1)[0]
    assert 'if (!busy) box.value = "";' in hero_submit
