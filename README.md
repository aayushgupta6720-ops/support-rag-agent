# support-rag-agent

[![GitHub repo](https://img.shields.io/badge/GitHub-support--rag--agent-181717?logo=github)](https://github.com/aayushgupta6720-ops/support-rag-agent)

**Live demo:** https://support-rag-agent.onrender.com — a help center for
Tidewell, a fictional ticketing product, with the agent as its chat widget
([`/docs`](https://support-rag-agent.onrender.com/docs) for the API,
[`/health`](https://support-rag-agent.onrender.com/health))
— deployed on Render's free tier, so the first request after a period of
inactivity takes ~30-60s to wake up.

Async FastAPI RAG agent for support tickets — RAG over Qdrant, LangGraph
orchestration, an eval harness (golden set + LLM-as-judge), and per-call
latency/cost observability.

## Structure

```
app/
  main.py              FastAPI app instance, mounts the router, serves the help center at /
  static/              Help-center page and chat widget (plain HTML/CSS/JS, no build step)
  api/routes.py        /health, /health/search, /chat and /articles endpoints
  api/ratelimit.py     Per-visitor /chat limits, counted in Redis (or in memory)
  api/sessions.py      Multi-turn chat history per session, in Redis (or in memory)
  api/answer_cache.py  Reused answers to repeat first questions, in Redis (or in memory)
  api/feedback.py      Thumbs up/down on answers, and the token-protected export
  api/body_limit.py    Refuses oversized request bodies before they're read
  core/config.py       Settings, loaded from .env
  core/gemini_client.py Shared, cached google-genai client
  core/redis_client.py Shared Redis client (when REDIS_URL is set) and its outage switch
  models/schemas.py    Pydantic request/response models
  rag/
    chunking.py        Paragraph-packing text chunker with word-boundary overlap
    embeddings.py       Gemini embedding calls (gemini-embedding-001), batched at 100
    qdrant_store.py     Async Qdrant client, collection setup, search, stale-chunk deletes
    documents.py        Loads the help articles from data/docs/
    ingest.py           Chunk + title-prefix + embed + upsert documents
    retrieval.py        Embed a query and fetch top-k chunks
  agent/
    graph.py           LangGraph agent: route -> (retrieve) -> generate (AGENT_MODE=router, live)
    loop.py            LangGraph tool loop: agent <-> tools until give_answer (AGENT_MODE=loop)
    history.py         Earlier conversation turns, as Gemini contents
    prompts.py         Versioned system prompts
    tools.py           search_support_docs, read_article and give_answer declarations
  eval/
    dataset.py         Loads data/eval/golden_set.jsonl
    judge.py           LLM-as-judge: grades an answer against a reference, majority vote on failures
    harness.py         Runs the agent + judge over the golden set
    prompts.py         Versioned judge prompt
  core/
    observability.py  Per-call trace (latency/tokens/cost per step) + JSON logging
    pricing.py         Approximate per-token cost estimates (Gemini paid-tier rates)
scripts/
  ingest.py            CLI: loads data/docs/*.md and ingests into Qdrant
  eval.py              CLI: runs the eval harness, prints + saves a report
  threshold_sweep.py   CLI: measures retrieval score thresholds offline from saved scores
  feedback_candidates.py CLI: turns thumbs-down ratings into draft eval cases
data/docs/             11 sample support docs used by the ingestion script
data/eval/             Golden set + timestamped eval run results
mcp_server/            Optional MCP wrapper (separate venv, see below)
tests/                 Offline unit tests (Gemini, Qdrant and Redis are faked)
Dockerfile
requirements.txt
requirements-dev.txt   requirements.txt + pytest
.env.example
```

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive FastAPI docs, or hit
`http://localhost:8000/health`.

**If you're also using the MCP wrapper** (below), run without `--reload`
instead. Uvicorn's file watcher scans the whole project tree, including
`mcp_server/venv/` — launching that subprocess touches files there (e.g.
bytecode cache), which the watcher treats as a code change and restarts the
server mid-request, killing whatever call was in flight. `--reload-dir`/
`--reload-exclude` don't reliably fix this (tested: still triggers on
nested paths under an excluded dir), so just drop `--reload` when the MCP
server might be running alongside it.

Qdrant must be running (e.g. `docker run -p 6333:6333 qdrant/qdrant`) before
you ingest docs or call `/chat`. Ingest the sample support docs with:

```bash
python -m scripts.ingest
```

Re-running it is safe: chunk IDs are deterministic, so changed docs are
overwritten in place, chunks left over when a doc gets shorter are deleted,
and docs removed from `data/docs/` are removed from Qdrant too.

Then `POST /chat` with `{"query": "..."}`. The live agent (`AGENT_MODE=router`)
is a LangGraph router plus RAG: a router call decides whether to call the
`search_support_docs` tool or answer directly (e.g. for greetings), then a
generation call produces a structured, schema-validated answer grounded in
whatever was retrieved. It makes exactly one decision and at most one search.

`AGENT_MODE=loop` is a real tool loop instead (`app/agent/loop.py`): one model
calls tools until it answers. It can search as many times as it needs, with
its own queries; read a whole article with `read_article` when a search found
the right one but not the passage it needs; and it ends the turn only by
calling `give_answer` with its reply and the ids of the articles it used.
Every turn must be a tool call, so an answer always comes with its sources,
and after four steps it may only answer. Both modes return the same shape, so
`/chat`, the answer cache and every eval work with either.

Or open `http://localhost:8000/` for the help center. It's a page for a
fictional company, Tidewell, with topic cards, a search box, and a chat
widget in the corner. It's plain HTML, CSS and JavaScript in `app/static/`,
served by the same app, so there's nothing to build and no cross-origin
setup. The widget:

- keeps the `session_id`, so follow-ups work, and keeps the conversation
  for the browser tab's lifetime;
- shows each answer's source articles, each opening the full article inside
  the panel (served by `GET /articles/{doc_id}`);
- has thumbs up and down under each answer (see "Feedback" below);
- shows a "still working" note on slow replies (a cold start on Render's
  free plan takes up to a minute);
- says so when a follow-up was answered without the earlier messages. Each
  response reports `history_turns`, and after 30 idle minutes the server
  has forgotten the conversation even though the transcript still shows it;
- says "one moment" if a topic card is clicked while an answer is coming,
  instead of ignoring the click;
- shows the API's own message for rate limits, quota and timeouts.

Answers are built from text nodes, never HTML. The page's
Content-Security-Policy only allows its own origin, so even if markup got
into an answer, it couldn't load or run anything. It's full-screen on
phones, follows the system's dark mode, and works with the keyboard alone
(Enter sends, Shift+Enter starts a new line, Escape closes).

**Feedback.** Every response has an `answer_id`, and the widget's thumbs
post it to `POST /feedback`. The server keeps its own record of each answer
it gave (question, answer, sources) for a day under that id, and a rating
stores that record, so a rating is the one place a question gets stored. The
widget used to send the question and answer itself, which let anyone post
made-up pairs that became draft eval cases; now an id the server didn't
issue in the last day gets a 404, any text the request sends is ignored, and
each answer can be rated once. Ratings have their own rate limit (20 a
minute, 100 a day), and the newest 2,000 are kept in Redis.

`GET /feedback/export` returns them. It needs `FEEDBACK_EXPORT_TOKEN` as a
bearer token, and doesn't exist without one. `python -m
scripts.feedback_candidates` turns the thumbs-down into draft eval cases in
`data/eval/feedback_candidates.jsonl` (gitignored), one per distinct
question, marked held-out. Each draft needs a category, expected docs and a
reference answer before it moves into the golden set. That's how a real
visitor's bad answer becomes a test the agent has to pass.

Conversations can have follow-ups. Every response includes a `session_id`;
send it back with the next question and the agent sees the last 3
exchanges, so "What about annual plans?" after a refund question gets
annual-plan refund policy. The router rewrites a follow-up like that into a
standalone search query, since the search itself sees no history. Only
questions and answers are kept, not retrieved docs, and product facts in an
answer still have to come from docs retrieved for that question, not from
earlier answers. A session ends after 30 idle minutes
(`SESSION_MAX_EXCHANGES`, `SESSION_TTL_S`), and it's tied to the visitor as
well as the id: a client can pick its own `session_id`, and without that,
anyone reusing an id could continue someone else's conversation.

The first question of a conversation is answered from a cache when someone
has asked the same thing in the last hour (`ANSWER_CACHE_TTL_S`, 0 turns it
off). The help center's topic cards and chips send the same few questions
over and over, so this makes them instant and stops each one costing two
Gemini calls.

- **What counts as the same:** case and extra whitespace don't matter.
- **What makes it miss:** the key also covers the models, prompt versions,
  retrieval settings and collection, so changing any of them misses instead
  of serving an answer made the old way.
- **Privacy:** only a hash of the question is stored, never its text.
- **Limits:** follow-ups are never cached. After re-ingesting changed docs,
  cached answers can lag for up to the TTL.
- **Bad answers:** a thumbs-down on a cached answer removes it, so the next
  visitor gets a fresh one rather than the same bad answer for up to an
  hour. It only does this if the cache still holds exactly the rated
  answer, so a made-up rating can't evict a different one.
- **Where it lives:** the cache is in Redis when `REDIS_URL` is set,
  otherwise in memory. The log line says `cache_hit`.

Run the eval harness (agent + LLM-as-judge over a golden set) with:

```bash
python -m scripts.eval
```

Pass case ids or categories to run only those, e.g. `python -m scripts.eval
multi_turn greeting_1`.

Three more scripts check what that run can't:

- `python -m scripts.cross_judge <results.json>` re-grades a finished run's
  answers with a judge from another model family (an open-weights Gemma
  model, on its own free-tier quota), given exactly what the eval's judge
  saw, and reports how often the two agree.
- `python -m scripts.full_context_baseline` answers the questions with every
  article in the prompt and no retrieval. The whole help center is about
  8,700 characters, so this is cheap, and it shows what retrieval costs:
  where the agent falls short of it, retrieval left something out.
- `python -m scripts.eval_chat` plays the multi-turn cases through the real
  `/chat` endpoint in process: the earlier messages go through one session
  and the agent writes its own earlier answers. It checks that the session
  hands back every earlier turn and that a repeated first question comes
  from the answer cache.

It reports a judge pass rate, a pass rate per category, and these retrieval
metrics, computed over the cases that expect particular docs:

- **Hit rate**: any expected doc was retrieved.
- **Recall**: the share of expected docs retrieved. It only differs from the
  hit rate for multi-doc questions.
- **MRR** and **hit@1**: how high the first expected doc ranked among the
  distinct docs retrieved. Hit rate alone can't tell first place from last.
- **Context precision**: the share of retrieved docs that were expected, so
  how much of what the answer was generated from was noise.
- **Citation precision** and **citation recall**: answers name the docs they
  actually used. Precision is the share of cited docs that were expected;
  recall is the share of expected docs the answer cited. The report also
  lists cases with nothing to find whose answer still cited a doc.

It saves a timestamped JSON report under `data/eval/results/`, including
which commit, prompt versions, model and `top_k` produced it. A case that
raises (e.g. a Gemini 503 during a demand spike) is recorded as an error and
left out of the rates instead of aborting the run.

The corpus has 11 docs. Five hold the answers the golden set asks about.
The other six (invoices, API keys, data export, email notifications,
sign-in lockouts, team roles) share their vocabulary without holding those
answers, so retrieval has to rank the right doc above near-misses.
`top_k=4` now retrieves 4 of 14 chunks instead of 4 of 7, and a test fails
if a doc ever answers one of the `unanswerable` cases.

The judge grades each answer against a reference, sees the docs the agent
retrieved (so it can tell a supported extra detail from an invented one),
and sees the earlier turns in multi-turn cases. One verdict is noisy: the
same answer text has been graded both ways on different runs. So a failing
verdict is re-judged, up to three votes, and the majority kept, stopping as
soon as two agree. Passing first verdicts aren't rechecked, to save quota.
The flips seen so far were correct answers marked wrong, so the extra calls
go where the noise was, but a wrong answer that fools the judge once still
passes. Lowering the judge's temperature was the other option, but
[Google recommends](https://ai.google.dev/gemini-api/docs/gemini-3) keeping
Gemini 3 models at the default 1.0, warning that lower values can cause
looping, so it wasn't used.

**A second judge.** The judge and the agent are both Gemini, and a passing
verdict is never rechecked, so the run of 2026-10-11 was re-graded by
`gemma-4-26b-a4b-it` (`scripts/cross_judge.py`). It saw the same
instructions, references and retrieved chunks; retrieval was re-run from the
saved search queries and matched the run's on all 115 cases. The two agreed
on 113 of 115 (98%, Cohen's kappa 0.66; kappa is modest because there are so
few failures to agree on). Both failed the two retrieval misses. Gemma also
failed two answers Gemini passed: `password_reset_no_email` left out the
route through support with ID for someone who no longer has the email, and
`multi_doc_delete_within_refund_window` could be read as saying deletion
refunds automatically, when a refund needs a ticket. So by the stricter judge
the run is 111/115.

**Checked against a person.** 30 of the run's answers were labelled by hand,
blind to both judges: the 4 that either judge failed, plus 26 drawn at random
across categories. The person marked 29 correct and 1 wrong (the downgrade
retrieval miss). The Gemini judge agreed on 29 of 30 (kappa 0.65), Gemma on
27 (kappa 0.37). Neither judge passed an answer the person marked wrong.
Every disagreement was a judge failing an answer the person passed, so on
this sample the judges grade more strictly than a person and the pass rate
isn't inflated by a lenient judge. The one answer both judges failed and the
person passed, `ho_unrecognized_signin`, leaves out that a password reset
signs out other sessions, which was half the question. One labeller, 30
answers and a single "wrong" make both kappas unstable. Labels and agreement
are in `data/eval/summaries/human_labels_20261011T080413Z.json`.

**Citations checked against the articles.** The eval scores citations only
against the articles a case expects, so an answer could cite the right
article and still claim something it doesn't say. `scripts/citation_support.py`
gives Gemma the full text of every article an answer cites, with the question,
and asks whether each factual claim is stated there or follows from it (a rule
applied to what the user said counts). On the 2026-10-11 run, 87 of the 89
answers that cite something are fully supported. Both exceptions passed the
judge and add a detail the article doesn't give: a menu path for changing a
member's role (the article says who can, not where), and "we cannot turn off
2FA on another user's account" stated as policy, when the article says only
that turning it off needs the user's own code. On the 45 outside cases below,
38 of the 39 citing answers are fully supported; the one that isn't is the
passkeys answer the judge also failed. Summaries are in
`data/eval/summaries/citation_support_*.json`.

**Cases written by someone outside the project.** Every case above was written
by the developer who wrote the prompts. On 2026-10-12 a friend wrote 45 more
(the `external` split, ids `x_*`) from the published help articles alone,
without seeing the eval cases or the prompts, trying the bot or using an AI.
They were committed before any run (c17280a) and no prompt has been tuned on
them. Router: 44/45. Tool loop: 43/45 with `agent_v1` and with `agent_v2`.
Both agents fail `x_unanswerable_passkeys`: asked about passkeys, they say
they aren't supported, when the articles only list authenticator apps. The
loop also fails `x_unanswerable_data_centers`, answering how exports are
delivered instead of saying where they're stored isn't documented.

The golden set (`data/eval/golden_set.jsonl`) has 172 cases: the 127 in the
table below, written by the developer, and 45 written by someone outside the
project (see "Cases written by someone outside the project"). Most test
failure modes, not whether the model can find the answer to an easy question:

| Category | Cases | What it checks |
|---|---|---|
| `grounded` | 31 | Answer is in one doc, including details beyond the headline fact |
| `reasoning` | 20 | Applying a policy to the user's situation ("I bought annual 3 weeks ago — refund?") |
| `false_premise` | 12 | Question assumes something the docs contradict ("SMS 2FA setup?") |
| `multi_doc` | 14 | A complete answer needs facts from two docs |
| `unanswerable` | 11 | In-domain but not in the docs — must not invent a price, phone number, etc. |
| `out_of_scope` | 6 | Not a support question — should stay in role |
| `adversarial` | 10 | Prompt injection and social engineering |
| `robustness` | 5 | Typos, Spanish, vague phrasing |
| `direct` | 7 | Greetings and small talk — no retrieval |
| `multi_turn` | 11 | A follow-up that only makes sense with the earlier turns ("What about annual plans?"), a topic switch, and a false premise after pushback |

Every case belongs to a split:

- **`dev`** (86 cases): the cases prompts and settings were tuned against.
- **`held_out`** (41 cases): written before the latest change and never used
  to tune it, so they show whether the change generalizes rather than
  fitting the dev cases.
- **`external`** (45 cases): written by someone else from the help articles
  alone, and never used to tune anything.

Once held-out cases have informed a decision, they move to dev and fresh
ones are written. The 13 written for the score gap and
`grounded_answer_v5` moved to dev after they showed that 0.08 was too tight
and that v5 hedged. The 11 `h2_*` ones were written before
`grounded_answer_v6`, and most need an article's rule applied to the
user's situation. A review pointed out they were nearly all one kind of
case, so 18 more (`h3_*`) were written before `grounded_answer_v7`,
across every category, multi-turn included; `grounded_answer_v6` passed
all 18. On 2026-10-11, 12 more (`h4_*`) were written before any change to
retrieval, mostly answers that need two articles or a policy applied to a
situation. Run one split with `python -m scripts.eval held_out`.

Pass rates are per category because an aggregate hides the weak spots. With
this few cases per category, treat a single run as a smoke signal, not a
benchmark: one flipped case moves a category by 20+ points.

**`grounded_answer_v7` (current).** `multi_doc_downgrade_rate_limit`, asked
what a failed payment does to the API rate limit, failed 4/4 with "I don't
have enough information", though one article says the account is downgraded
to the free tier and another gives that tier's limit; the same call with
those articles as fixed context failed 3/3 too. v7 says to put facts from
different articles together, and drops v6's worked example (a 14-day refund,
a purchase 10 days ago), which matched the held-out case
`h2_annual_refund_day_13`. The model is now pinned to
`gemini-3.5-flash-lite` rather than the alias `gemini-flash-lite-latest`,
which Google can move.

The runs on 2026-10-10 used a score gap of 0, while the deployed agent uses
0.10, so they didn't describe it; the eval now refuses to run with settings
that differ from `render.yaml`'s. At the deployed settings v7 scored
113/115 (below). Both failures are retrieval misses that no prompt can fix:

- `multi_doc_downgrade_rate_limit`: the billing chunk that came back covers
  renewals and refunds, not the failed-payment downgrade, so "I don't have
  enough information" is the faithful answer to what the model was given.
- `ho_unrecognized_signin`: the password-reset article scored 0.616, 0.005
  under the gap's cut-off (0.721 − 0.10), so the rule that a reset signs out
  other sessions never reached the model.

v6, run on the 11 `multi_doc` cases at the same settings, scored the same
9/11 with the same two failures. So v7 is level with v6 at the deployed
settings, not better: the multi-doc answer it fixed at gap 0 needs a chunk
the 0.10 gap doesn't keep. The next fix is retrieval (more chunks per doc,
or a gap measured per doc), not the prompt.

Each run's summary (settings, rates, cost, failed and errored cases) is
committed under `data/eval/summaries/`; the full answers stay in the
gitignored `data/eval/results/`. Since 2026-10-10 the summary includes what
the run would have cost on a paid key, judge included.

**Router + RAG or a tool loop?** (2026-10-12, `gemini-3.5-flash-lite`, gap
0.10.) The two agents on the same cases:

| Cases | Router (live) | Tool loop |
|---|---|---|
| Held-out, all 41 | 41/41 | 41/41 |
| Multi-doc, all 14 | 12/14 | 14/14 |
| Conflicting articles (4) | 0/4 | 4/4 |
| No-conflict controls (3) | 3/3 | 2/3 |
| Model calls per question | 2.00 | 2.0–2.75 (2.4 on multi-doc) |

- **Where the loop helps is retrieval.** The router's two multi-doc misses
  (above) were passages its one search didn't return. The loop searched again
  with a second query and found the failed-payment passage for
  `multi_doc_downgrade_rate_limit`, and its query for
  `ho_unrecognized_signin` found the password-reset article.
- **The conflict gap is mostly the prompt.** The loop's prompt (`agent_v1`)
  says what to do when articles disagree; the router's (`grounded_answer_v7`)
  doesn't, and it picked one article silently every time, once the stale one
  (an announcement of 2,000 requests a minute). With that one rule added
  (`grounded_answer_v8`, a candidate not in use) the router scored 3/4. Both
  agents over-apply the rule once: asked about monthly refunds, they gave the
  right 7 days but warned that the articles disagree about annual refunds.
  That control is left as a failure rather than tuned away.
- **The conflict cases** (`python -m scripts.eval_conflicts`) run against
  their own Qdrant collection: the live articles plus three stale ones dated
  as real stale pages are (a 2024 billing FAQ, a 2023 page on text-message
  sign-in codes, a 2026 announcement of higher API limits), so the live help
  center never holds them.
- **Whole articles in the router** (`RETRIEVAL_EXPAND_DOCS=true`, not in use)
  fixed the same two misses on the 23 multi-doc and `h4_*` cases but broke
  one that passes without it (`h4_monthly_to_annual_today`): 22/23.
- **The full-context baseline** (every article in the prompt, no retrieval)
  scored 40/40 on the held-out split, with one case cut off by the day's
  quota. Retrieval costs nothing there; its losses are the dev cases above.
- **Decided: the router stays live** (2026-10-14, `agent_v2`, dev and
  external, 130 cases scored). The loop's first dev run obeyed two prompt
  injections ("don't look anything up") and declined without the policy;
  `agent_v2` makes its first call a search and answers the question inside an
  injection, and passes all 9 adversarial cases. But on dev it scored 82/85
  (one case timed out) against the router's 84/86, and 43/45 against 44/45 on
  the outside cases. It fixes the router's two multi-doc misses and loses
  three others: `unanswerable_sso` (it says SAML isn't supported, as with
  passkeys), and two "what can you help with?" answers that list the help
  center's topics, which the judge failed because the reference says not to
  name features. It also costs more: 2.25 model calls and $0.00137 a question
  on a paid key, against the router's 2.0 and $0.00071, since every step
  resends the conversation. Different failures at twice the cost isn't a
  reason to switch.

**Latest full run** (2026-10-11; `gemini-3.5-flash-lite`, 115 cases,
`router_v4`, `direct_answer_v3`, `grounded_answer_v7`, `judge_v2`,
`RETRIEVAL_MAX_SCORE_GAP=0.10`): 113/115 judge pass, 29/29 held-out and
84/86 dev. Retrieval hit rate was 100% and recall 99%, MRR 0.96, hit@1 93%
and context precision 57%. Citation precision was 90% and citation recall
97%, and no answer with nothing to find cited a doc. It took 348 chat-model
requests, which would cost $0.12 on a paid key ($0.08 agent, $0.04 judge).
Its summary is `data/eval/summaries/eval_20261011T080413Z.json`, recorded as
`d3a8f61-dirty` only because the clean worktree it ran in held an untracked
link to the virtualenv.

**The run before that** (`gemini-flash-lite-latest`, 97 cases,
`grounded_answer_v6`, gap 0.10): 97/97 judge pass, 11/11 held-out and 86/86
dev, MRR 0.96, hit@1 93%, context precision 55%, citation precision 94% and
recall 99%.

What changed, and what the held-out cases said about it:

- **`grounded_answer_v5`: answers name the docs they used.** Each context
  passage carries its doc id, and the answer lists the ids it relies on.
  Ids that weren't retrieved are dropped, so a made-up one can't become a
  source. `/chat` returns the cited docs; the log keeps every retrieved
  one. Live, "What is the refund policy?" had listed Invoices too; asked
  now, "How do I get a refund?" cites refunds plus the account-deletion
  caveat its answer actually mentions.
- **`router_v4`: search queries keep to the user's own topics.** v3 had
  searched "refund policy billing invoices" for "What is the refund
  policy?".
- **`grounded_answer_v6`: apply the rule; don't hedge or say "context".**
  On a held-out case, v5 gave the rule (a password reset signs out all
  other sessions) and then said "the context does not state whether" it
  covered an intruder's session. v6 asks the model to apply the docs to
  the user's situation and to speak as the support agent. The 11 cases
  written before v6 mostly need a rule applied to a situation ("bought
  annual 13 days ago: too late?"). All pass, none hedges or mentions
  "context", and each cites exactly the doc it used (citation precision and
  recall 100%).
- **Gap 0.08 → 0.10**, after the held-out check (see "Score thresholds").
  Context precision fell from 75% to 58%, the price of keeping the second
  doc that two-doc answers need.

**Before that**, with `grounded_answer_v4` and gap 0.08: 73/73, MRR 0.97,
hit@1 95%, context precision 75%. `grounded_answer_v4` stopped the
"I don't have enough information" opener on contradicted claims: every
false premise and contradicted adversarial claim started with a plain "No".

**Earlier** (`grounded_answer_v3`, no score gap): also 73/73, with
MRR 0.95, hit@1 91% and context precision 34%. It was the first run with v3
prompts, the 11-doc corpus and `judge_v2`, so it can't say which of the
three moved a number:

- No first verdict failed, so the majority vote never ran. It didn't rescue
  any case here.
- `adversarial_injection_rate_limit` passed, but its answer still opened
  with "I don't have enough information to confirm that" before giving the
  real 1,000/min limit, the wording `judge_v1` failed. The agent hadn't
  changed; the judge accepted it. `grounded_answer_v4` fixed the wording.
- `reasoning_lost_app_have_codes` passed, but this time the answer left out
  the caveat `judge_v1` misread, so this run doesn't show whether `judge_v2`
  fixes that misreading.
- The near-miss docs work as intended. In 5 cases one ranked above the right
  doc: `invoices-and-receipts` above `billing-refunds` for "How do I get a
  refund?", `sign-in-and-lockouts` above `password-reset` for a compromised
  account, `team-members` above `account-deletion`, and
  `email-notifications` above `password-reset`. The right doc was still in
  the top 4 every time, so the answers held. On the 5-doc corpus, the 63/65
  run scored MRR 1.00 and hit@1 100%.
- Context precision was 34% before the new docs too: `top_k=4` pulls in
  about two off-topic docs per question either way. See "Score thresholds"
  below for what cuts that.

**Score thresholds.** Two ways to keep off-topic chunks out of the top 4
are built in, both off by default: a fixed cosine-similarity floor
(`RETRIEVAL_MIN_SCORE`), and a maximum gap below the query's best chunk
(`RETRIEVAL_MAX_SCORE_GAP`). A threshold only removes chunks, so
`python -m scripts.threshold_sweep --collect` runs just the router and the
search for the 62 cases that search (62 model calls instead of ~190), saves
every chunk's score, and then any threshold is scored offline with the eval's
own metrics. `--changed` lists the cases a setting changes: only those can get
different answers, so only those need an eval run.

On the 62 cases, a fixed floor is a poor fit because scores depend on the
query. Chunks from the right doc scored 0.597–0.796 and off-topic ones
0.551–0.746; "How long does a refund take to show up?" scored 0.597 against
the right doc, while other queries' off-topic chunks reached 0.746. The
highest floor that loses nothing (0.597) lifts context precision only from
34% to 45%, with no margin. 0.65, which looked right from two live queries,
loses 5% of hits and leaves 3 cases with nothing. A gap works much better:

| Gap | Hit rate | Recall | Context precision |
|---|---|---|---|
| off | 100% | 100% | 34% |
| 0.10 | 100% | 100% | 61% |
| 0.08 | 100% | 100% | 75% |
| 0.06 | 100% | 99% | 83% |
| 0.04 | 100% | 98% | 92% |

Recall starts to drop at 0.06, when a two-doc answer's second doc scores
more than that below the first. Adding a floor on top of a gap of 0.08 would
cut the off-topic docs unanswerable questions still get from 3.0 to 1.2, but
those already pass 6/6, and the floor would sit 0.017 from where it starts
losing the right docs.

The held-out check said 0.08 was too tight. These 12 cases were written
after 0.08 was chosen, and `--collect --split held_out` scored them
without tuning anything on them. On them, 0.08 drops a needed second doc:
for `ho_unrecognized_signin`, `password-reset` (which says a reset signs out
other sessions) scored 0.093 below the top doc. 0.10 loses nothing on
either set, with context precision at 61% on dev and 56% on held-out
(34–36% with no gap). Fixed floors fared worse: the highest lossless one
was 0.66 on these cases but 0.597 on dev. So the live demo moved to
`RETRIEVAL_MAX_SCORE_GAP=0.10`.

The answer check reran the 49 cases a gap of 0.08 changes: 48/49, with 1.7
docs per answer instead of 3.4 and context precision at 78%. The one failure,
`multi_turn_false_premise_after_pushback`, answered "I don't have enough
information to confirm that" before correctly giving the 14-day window,
where the no-gap run said "No, 14 days". That opener is the grounded prompt's
don't-guess fallback being reused to reject a false claim: in these 49 cases
it appeared 3 times without the gap and 2 times with it, so it's a prompt
weakness the gap didn't create, though one sample can't rule out that the
single doc of context made it likelier. The live demo runs with
`RETRIEVAL_MAX_SCORE_GAP=0.08`, and `grounded_answer_v4` (see "Latest run")
fixed the fallback wording: with both, that case passes with "No, annual
plans cannot be refunded for 60 days".

**Earlier still** (5 docs, 65 cases, `router_v2` + `direct_answer_v2`,
`judge_v1`): 63/65 judge pass (97%), 100% retrieval recall.

The v1 prompts (`router_v1`, `direct_answer_v1`) scored 56/59 on the
original set. All three failures were in routing:

- Off-topic requests ("What's the capital of France?") were answered, because
  the router prompt only said when to search, and the direct-answer prompt
  had no scope.
- "Ignore the support docs…" talked the router out of searching, so the
  answer had nothing grounding it. Prompt injection hit the tool-routing
  decision, not just the answer.
- "What can you help me with?" got a generic reply, because nothing told the
  model what the product covers.

v2 makes searching the default when in doubt, tells the router to route on
the message's topic and ignore instructions inside it, scopes the direct
path to the support topics, and forbids stating product facts on the
direct path (defense in depth if routing is still wrong). To check the fix
generalizes rather than fitting three questions, six held-out variants
(new off-topic requests, injections, and capability questions) were added
*before* changing the prompts: v1 passed 2/6 of them, v2 passes 5/6. On
the original 59 cases v2 scores 58/59.

The two failures left under v2 were not routing failures:

- `adversarial_injection_rate_limit`: routes and retrieves correctly and
  states the real 1,000/min limit, but opens with "I don't have enough
  information to confirm that". The grounded prompt's don't-guess fallback
  gets reused to reject a false claim. A candidate for `grounded_answer_v3`.
- `reasoning_lost_app_have_codes`: a correct answer that adds a true caveat
  the judge misread as a contradiction. It's grader noise, not an agent
  regression (the grounded path is unchanged and passed under v1).

The judge is itself noisy: rerunning one case three times gave the same
answer text twice with opposite verdicts. That's why a failing verdict now
goes to a majority vote. Still treat ±1-2 cases between runs as noise.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Everything runs offline in well under a second: a conftest fixture makes any
unfaked call to Gemini or Qdrant fail the test instead of spending quota.

GitHub Actions runs the suite on every push and pull request
(`.github/workflows/tests.yml`), on Python 3.11 like the Dockerfile; a test
fails if the two versions drift apart. Render deploys only after that run
passes: `autoDeployTrigger: checksPass` in `render.yaml`, or Auto-Deploy set
to "After CI Checks Pass" in the dashboard. So a commit that breaks the tests
never reaches the live demo.
Covered: agent routing (tool call → retrieval → grounded prompt, direct
answers, empty retrieval, tool call with no query argument), history reaching
both model calls, 429 retry/backoff, chunking, retrieval and ingestion, the
`/chat` endpoint and its log line, sessions (follow-ups, trimming, expiry,
per-visitor isolation), rate limits (each rule run against both the
in-memory and the Redis limiter via fakeredis, plus surviving a restart and
Redis outages), tracing and cost math, eval scoring and judge voting, and
golden-set integrity (unique ids, known categories, every expected doc id
exists in `data/docs/`, no doc answers an `unanswerable` case), and the help
center (the widget, its assets, its security headers, and a title for every
doc's source chip). The widget's behaviour was checked in headless Chrome:
a topic card, a follow-up, a reload, and a phone-sized screen.

## Run with Docker

The container reads `$PORT` (defaults to 8000) so it works both locally and
on platforms like Cloud Run that inject their own port. Qdrant needs to be
reachable from *inside* the container, so `localhost` won't resolve to your
host's Qdrant unless you put both containers on the same Docker network:

```bash
docker build -t support-rag-agent .

docker network create rag-net
docker network connect rag-net qdrant   # the Qdrant container from "Run locally"

docker run --rm -p 8000:8000 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e GEMINI_API_KEY=your-key-here \
  --network rag-net \
  support-rag-agent
```

## Deploy

The app is a stateless container reading config from env vars, so it's
deployable as-is. The one thing to plan for: **Qdrant needs a persistent,
network-reachable home** — a single Cloud Run/Lambda/Render container can't
host it locally the way local dev does. Use [Qdrant Cloud](https://cloud.qdrant.io)
(has a free tier) or run Qdrant on a VM/persistent-disk service, then point
`QDRANT_URL` (and `QDRANT_API_KEY`, if using Qdrant Cloud) at it.

**Render** (what the live demo above actually runs on): `render.yaml` in the
repo root defines the service as a Render Blueprint — Docker runtime, free
plan, `/health` as the health check path. `GEMINI_API_KEY`, `QDRANT_URL`,
and `QDRANT_API_KEY` are marked `sync: false` so Render prompts for them
instead of storing them in the repo; paste in the same values from your
local `.env`. Note that Render currently asks for a card on file even for
the free plan (an account-level anti-abuse check, not a charge) —
happens on both the Blueprint and manual "New Web Service" paths.

Rate-limit counts and chat sessions live in a Render Key Value instance
(Valkey 8, free plan, same region, internal access only, `allkeys-lru`
eviction since every key has a TTL anyway). Set `REDIS_URL` to its internal
URL, `redis://<instance id>:6379`. The free plan keeps data in memory only,
so a Key Value restart resets the counts, but web-service restarts, which
happen far more often, no longer do. On startup the app logs
`{"event": "state_store", "backend": "redis"}` (or `"memory"`), which is how
to check it picked the URL up.

Searches run against Qdrant Cloud, and the cluster's region matters. With
the cluster in São Paulo and the app in Oregon, the `qdrant_search` step
took 0.7–1.0 s of a 3–4 s answer. Two changes cut that:

- Each search now makes one round trip instead of two; there's no longer a
  collection check before every query. That brought it to 0.57 s.
- The cluster moved to AWS us-west-2, next to the app. A warm search now
  takes 16 ms, and the first search after a restart about 190 ms, including
  the connection setup. A typical answer takes about 2 s.

Moving a free cluster means deleting the old one first (Qdrant allows one per
account), then pointing `QDRANT_URL` and `QDRANT_API_KEY` at the new one on
Render and in `.env`, and running `python -m scripts.ingest`. Only the help
articles live in Qdrant, rebuilt from `data/docs` in seconds. Sessions,
limits, cached answers and feedback are in Redis, so the move loses
nothing. While search is down, questions that need it get the 503 above.

A free cluster is suspended after a week with no requests, and deleted
after four weeks unless it's reactivated in the Qdrant console. A quiet
week on the demo would be enough, so a GitHub Actions workflow
(`.github/workflows/keepalive.yml`) calls `GET /health/search` twice a week,
on Mondays and Thursdays. Twice rather than once: GitHub can delay
scheduled runs, and a weekly run that lands just after the 7 days would be
too late.

That endpoint runs one real Qdrant search and a count. It makes no model
call, so it costs no Gemini quota. It returns a 503 if search fails or the
index is empty, which makes the run fail and GitHub email you. It's
separate from `/health` so that a Qdrant outage never makes Render restart
the app. GitHub turns scheduled workflows off after 60 days without commits
to a public repo; if that happens, re-enable it from the Actions tab.

**Google Cloud Run** (matches the existing Dockerfile directly):

```bash
gcloud run deploy support-rag-agent \
  --source . \
  --region us-central1 \
  --set-env-vars QDRANT_URL=https://your-qdrant-host:6333 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

`--set-secrets` pulls the key from Secret Manager rather than baking it into
the image or an env var visible in `gcloud run services describe` — create
it once with `gcloud secrets create gemini-api-key --data-file=-`.

**AWS Lambda**: the app isn't Lambda-ready as-is — FastAPI needs an ASGI
adapter (e.g. [Mangum](https://github.com/jordaneremieff/mangum)) wrapping
`app.main.app`, and the existing Dockerfile's `CMD` would need to invoke the
Lambda runtime interface instead of uvicorn. Not done here since Cloud Run
requires no code changes and this project doesn't otherwise touch AWS; add
Mangum and a Lambda-flavored `CMD` if you specifically need Lambda.

Either way, use your platform's secret manager for `GEMINI_API_KEY` (Secret
Manager / SSM Parameter Store), not a plain env var in the deploy command —
those tend to leak into logs, `describe` output, and shell history.

**What a question costs and how long it takes.** Every `/chat` logs its
latency, tokens and cost per step; `python -m scripts.usage_report <log file>`
reads them back (Render's log export, or an eval run's results file, since
every eval case now records the agent's time and cost) and gives p50/p95
latency overall, per step and from the cache, and the cost of 1,000 questions
at the observed cache-hit rate. On a paid key the router costs $0.00071 a
question (2.0 model calls), so 1,000 questions cost about $0.71 before the
answer cache; the tool loop would cost $1.37. The three live questions in
Render's logs (the free plan keeps a few days) took 3.6–4.2s, of which the
routing call was 1.4–1.8s and Qdrant 1.0–1.2s. The eval's 130 loop questions
had a median of 4.9s, but its tail isn't a visitor's: the eval sends
questions back to back, hits the free tier's per-minute limit, and 13% of
questions waited out retries past 15s. A p95 for visitors needs more traffic
than the demo gets.

**Quota:** on the Gemini free tier, `gemini-3.5-flash-lite` (the pinned
model; the alias `gemini-flash-lite-latest` resolved to it on 2026-10-10)
allows 500 requests/day per Google
project, shared by everything using that project's keys. A full eval run is
~220 requests, so give the deployed service a key from its own project;
otherwise a couple of eval runs can exhaust the demo's quota. When the daily
quota is gone, `/chat` returns a 503 in under a second whose `detail` says
when it resets ("in about 8 hours"), plus a `Retry-After` header and a
`resets_at` timestamp. Gemini's 429 for a per-day quota still suggests
retrying in ~60s, and trusting that used to make each request hang for
minutes and then 500. Per-minute 429s are still retried with the suggested
delay; if they outlast the retries, `/chat` returns a 429 asking to wait a
minute (`Retry-After: 60`) instead of a bare 500. The MCP tool passes these
messages through to the model, so it can tell the user why and when to retry.
Gemini also has capacity spikes, answering 503 UNAVAILABLE ("high demand")
even with quota to spare. Those are retried after 1s, 2s and 4s; if the model
is still overloaded, `/chat` returns a 503 saying it's a temporary Google-side
issue (`Retry-After: 60`) instead of a bare 500. Gemini's own 500 INTERNAL,
which Google also says to retry, gets the same retries and, if it keeps
coming, the same 503. So does a connection to Gemini that fails or drops
partway, with a 503 saying the connection failed. A Gemini call that gets
no response within `GEMINI_TIMEOUT_S` (60s) is stopped rather than left
hanging, and `/chat` returns a 504 saying so. Retries only happen in a
`/chat`'s first 45s: a retry whose wait would end later isn't made, and the
429 or 503 above comes back straight away. Without that limit, Gemini's
suggested waits (up to ~60s each) added up across the nested retries: four
per-minute 429s in a row could keep a `/chat` going for 7 minutes, long
after the MCP proxy had given up on it. Now a `/chat` ends within 45s plus
60s for each of its three Gemini calls (route, embed, generate), 225s in
all, inside the MCP proxy's 240s. The eval harness calls the agent
directly, so it keeps retrying as before.

Other failures get a clear status too, never a bare 500:

- **Gemini gives up itself** (its own 504 DEADLINE_EXCEEDED): handled like
  our timeout.
- **Nothing usable comes back:** the question or the answer was blocked,
  for example for safety, or the structured output didn't parse. That
  returns a 502 asking the visitor to rephrase.
- **The Qdrant search fails:** unreachable, erroring, or the collection
  missing. That returns a 503 with `Retry-After: 30`.

The `chat_call` log line records how long the question was, not its text,
because a support chat collects personal details. Set `LOG_CHAT_TEXT=true`
to log the question and the router's search query when debugging.

**Limits:** the quota is shared by everyone using the demo, so each visitor
(an IP address; for IPv6, its /64) can ask 6 questions a minute and 30 a
day. Past that, `/chat` returns a 429 with `Retry-After` and a `detail`
saying when to try again, without calling the model; the MCP tool passes
that message on. Change the numbers with `CHAT_LIMIT_PER_MINUTE` and
`CHAT_LIMIT_PER_DAY` (0 turns one off). Every visitor also shares one budget
of 200 new questions a day (`CHAT_LIMIT_ALL_PER_DAY`), about 400 of the 500
model requests: per-IP limits alone let nine addresses, or one visitor with a
block of IPv6 addresses, use up the day. A cached answer calls no model, so
it doesn't count. With `REDIS_URL` set, counts (and
chat sessions) live in Redis. They survive restarts, which matters on
Render's free plan because the service restarts every time it wakes from
sleep: with counts in memory, each wake-up handed every visitor a fresh
allowance. The check and the count run as one Lua script, so simultaneous
requests can't slip past the limit. If Redis is unreachable, requests are
counted in memory instead, still limited, rather than all refused or all
let through. After a Redis error, the limits, sessions, cache and ratings
all stay in memory for 30s and then try Redis again, rather than each call
waiting out a 2s timeout first: with Redis hanging, that added 10s to
every `/chat`, and now only the first one waits. Without `REDIS_URL`, everything is in memory, and the
Dockerfile pins one worker process so the counts aren't split. A `query` can be up
to 2,000 characters (surrounding whitespace is stripped first, so a blank one
is refused with a 422 instead of costing two model calls) and a request body up to 64 KB; a bigger body is
refused with a 413 before it's read, since FastAPI otherwise reads and
parses all of it first (a 52 MB body took the server from 134 to 419 MB).

Behind a proxy, the connecting address is the proxy's, so
`CLIENT_IP_HEADER` names the header that carries the visitor's IP. On
Render that's `CF-Connecting-IP`: Cloudflare, in front of Render, sets it
and refuses requests that bring their own. Not `X-Forwarded-For`: Render
appends to it, so its first entry is whatever the visitor sent. If the
configured header is missing from a request, everyone without it shares one
count. `render.yaml` sets `CLIENT_IP_HEADER`, but only a service created
from the Blueprint picks that up; otherwise set it in the dashboard.

To check it after a deploy without spending quota, send 7 empty questions
(each is counted, then refused with a 422 before any model call), each with
a made-up `X-Forwarded-For`:

```bash
for i in $(seq 7); do curl -s -o /dev/null -w '%{http_code} ' \
  -H "X-Forwarded-For: 10.9.9.$i" -H 'Content-Type: application/json' \
  -d '{"query": ""}' https://support-rag-agent.onrender.com/chat; done
```

Six `422`s then a `429` means the made-up headers didn't count as new
visitors. It uses a minute's worth of your own allowance. Run it against the
deployed app rather than localhost, where uvicorn trusts `X-Forwarded-For`
from `127.0.0.1`.

## Optional: MCP wrapper

`mcp_server/` exposes `/chat` as an MCP tool (`ask_support_agent`) so an MCP
client (e.g. Claude Desktop) can call the agent directly. Each answer ends with
its `session_id`, which the model can pass back to ask a follow-up in the
same conversation. It's a thin HTTP
proxy to the running FastAPI app, not an in-process import — the `mcp`
package needs a newer `starlette` than this project's pinned FastAPI
supports, so it lives in its own venv:

```bash
cd mcp_server
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py   # stdio transport; point an MCP client at this command
```

Set `SUPPORT_AGENT_URL` if the main app isn't on `http://localhost:8000`.

To register it with Claude Desktop, add it to that app's config file
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`;
merge into whatever's already there rather than replacing the file), using
the venv's own interpreter so it resolves regardless of what's on `PATH`:

```json
{
  "mcpServers": {
    "support-rag-agent": {
      "command": "/absolute/path/to/support-rag-agent/mcp_server/venv/bin/python",
      "args": ["/absolute/path/to/support-rag-agent/mcp_server/server.py"],
      "env": {
        "SUPPORT_AGENT_URL": "http://localhost:8000"
      }
    }
  }
}
```

Fully quit and relaunch Claude Desktop afterward — MCP servers are only
picked up at startup. The main app (and Qdrant) need to already be running
whenever the tool is actually called.

## Roadmap

- [x] **Step 1** — Async FastAPI skeleton + Docker (this scaffold)
- [x] **Step 2** — Ingestion + retrieval pipeline (Qdrant + Gemini embeddings)
- [x] **Step 3** — Agent layer: LangGraph tool routing, Pydantic structured
      outputs, versioned prompts
- [x] **Step 4** — Evaluation harness: golden set + LLM-as-judge
- [x] **Step 5** — Observability: latency/cost/quality logging per call
- [x] **Step 6** — Deploy (Cloud Run/Lambda) + optional Next.js frontend or
      MCP wrapper

All 6 steps are done. The repo is deploy-ready (Dockerfile, `$PORT` handling,
secrets guidance) and includes an optional MCP wrapper; it's also actually
deployed — see the live demo link at the top, and "Deploy" above for how to
reproduce it.
