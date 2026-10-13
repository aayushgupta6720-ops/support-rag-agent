import json

from scripts import citation_support


def test_the_prompt_carries_every_cited_article_and_the_answer():
    prompt = citation_support.citation_prompt("How long?", "Refunds take 5 days.", {"billing-refunds": "# Refunds\n5 days."})
    assert "ARTICLE billing-refunds:\n# Refunds\n5 days." in prompt
    assert prompt.index("CITED ARTICLES") < prompt.index("QUESTION:\nHow long?") < prompt.index("ANSWER:\nRefunds take 5 days.")
    assert prompt.endswith(citation_support.REPLY_FORMAT)


def test_verdicts_are_read_from_fenced_replies_and_bad_ones_are_none():
    reply = '```json\n{"supported": false, "unsupported": ["refunds take 5 days"]}\n```'
    assert citation_support.verdict_from(reply) == (False, ["refunds take 5 days"])
    assert citation_support.verdict_from('{"supported": true}') == (True, [])
    assert citation_support.verdict_from('{"supported": "yes"}') is None
    assert citation_support.verdict_from("no json") is None


def test_cited_articles_are_read_from_the_docs_and_unknown_ids_are_refused():
    texts = citation_support.cited_articles(["password-reset", "password-reset"])
    assert list(texts) == ["password-reset"] and texts["password-reset"].strip()
    assert citation_support.cited_articles(["password-reset", "no-such-article"]) is None


def test_the_summary_counts_only_parsed_verdicts():
    rows = [
        {"id": "a", "supported": True},
        {"id": "b", "supported": False, "unsupported_claims": ["x"]},
        {"id": "c", "supported": None},
    ]
    summary = citation_support.summarise("eval_x.json", {}, "gemma", rows)
    assert summary["citing_answers"] == 3 and summary["unparsed"] == ["c"]
    assert summary["support_rate"] == 0.5 and [r["id"] for r in summary["unsupported"]] == ["b"]
    json.dumps(summary)
