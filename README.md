# support-rag-agent

[![GitHub repo](https://img.shields.io/badge/GitHub-support--rag--agent-181717?logo=github)](https://github.com/aayushgupta6720-ops/support-rag-agent)

**Live demo:** https://support-rag-agent.onrender.com ([`/health`](https://support-rag-agent.onrender.com/health), `POST /chat`)
— deployed on Render's free tier, so the first request after a period of
inactivity takes ~30-60s to wake up.

Async FastAPI RAG agent for support tickets — RAG over Qdrant, LangGraph
orchestration, an eval harness (golden set + LLM-as-judge), and per-call
latency/cost observability.

## Structure

```
app/
  main.py              FastAPI app instance, mounts the router
  api/routes.py        /health and /chat endpoints
  api/ratelimit.py     Per-visitor /chat limits, counted in Redis (or in memory)
  api/sessions.py      Multi-turn chat history per session, in Redis (or in memory)
  api/body_limit.py    Refuses oversized request bodies before they're read
  core/config.py       Settings, loaded from .env
  core/gemini_client.py Shared, cached google-genai client
  core/redis_client.py Shared Redis client, when REDIS_URL is set
  models/schemas.py    Pydantic request/response models
  rag/
    chunking.py        Paragraph-packing text chunker with word-boundary overlap
    embeddings.py       Gemini embedding calls (gemini-embedding-001), batched at 100
    qdrant_store.py     Async Qdrant client, collection setup, search, stale-chunk deletes
    ingest.py           Chunk + title-prefix + embed + upsert documents
    retrieval.py        Embed a query and fetch top-k chunks
  agent/
    graph.py           LangGraph agent: route -> (retrieve) -> generate
    history.py         Earlier conversation turns, as Gemini contents
    prompts.py         Versioned system prompts
    tools.py           search_support_docs tool declaration
  eval/
    dataset.py         Loads data/eval/golden_set.jsonl
    judge.py           LLM-as-judge: grades an answer against a reference, majority vote on failures
    harness.py         Runs the agent + judge over the golden set
    prompts.py         Versioned judge prompt
  core/
    observability.py  Per-call trace (latency/tokens/cost per step) + JSON logging
    pricing.py         Approximate per-token/per-char cost estimates
scripts/
  ingest.py            CLI: loads data/docs/*.md and ingests into Qdrant
  eval.py              CLI: runs the eval harness, prints + saves a report
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

Then `POST /chat` with `{"query": "..."}`. A LangGraph agent routes the query:
a router call decides whether to call the `search_support_docs` tool or
answer directly (e.g. for greetings), then a generation call produces a
structured, schema-validated answer grounded in whatever was retrieved.

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

Run the eval harness (agent + LLM-as-judge over a golden set) with:

```bash
python -m scripts.eval
```

Pass case ids or categories to run only those, e.g. `python -m scripts.eval
multi_turn greeting_1`.

It reports a judge pass rate, a pass rate per category, and these retrieval
metrics, computed over the cases that expect particular docs:

- **Hit rate**: any expected doc was retrieved.
- **Recall**: the share of expected docs retrieved. It only differs from the
  hit rate for multi-doc questions.
- **MRR** and **hit@1**: how high the first expected doc ranked among the
  distinct docs retrieved. Hit rate alone can't tell first place from last.
- **Context precision**: the share of retrieved docs that were expected, so
  how much of what the answer was generated from was noise.

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

The golden set (`data/eval/golden_set.jsonl`) has 73 cases. Most of them test
failure modes, not whether the model can find the answer to an easy question:

| Category | Cases | What it checks |
|---|---|---|
| `grounded` | 23 | Answer is in one doc, including details beyond the headline fact |
| `reasoning` | 5 | Applying a policy to the user's situation ("I bought annual 3 weeks ago — refund?") |
| `false_premise` | 6 | Question assumes something the docs contradict ("SMS 2FA setup?") |
| `multi_doc` | 5 | A complete answer needs facts from two docs |
| `unanswerable` | 6 | In-domain but not in the docs — must not invent a price, phone number, etc. |
| `out_of_scope` | 4 | Not a support question — should stay in role |
| `adversarial` | 7 | Prompt injection and social engineering |
| `robustness` | 3 | Typos, Spanish, vague phrasing |
| `direct` | 6 | Greetings and small talk — no retrieval |
| `multi_turn` | 8 | A follow-up that only makes sense with the earlier turns ("What about annual plans?"), a topic switch, and a false premise after pushback |

Pass rates are per category because an aggregate hides the weak spots. With
this few cases per category, treat a single run as a smoke signal, not a
benchmark: one flipped case moves a category by 20+ points.

**Latest run** (`gemini-flash-lite-latest`, 73 cases, 11 docs, `router_v3`,
`direct_answer_v3`, `grounded_answer_v3`, `judge_v2`): 73/73 judge pass,
100% retrieval hit rate and recall, MRR 0.95, hit@1 91%, context precision
34%, no errored cases. All 8 multi-turn cases pass, including the topic
switch and the false premise after pushback.

Three things changed at once (the prompts, the corpus, and the judge), so
this run can't say which one moved a number. Read it with that in mind:

- No first verdict failed, so the majority vote never ran. It didn't rescue
  any case here.
- `adversarial_injection_rate_limit` passed, but its answer still opens
  with "I don't have enough information to confirm that" before giving the
  real 1,000/min limit, the wording `judge_v1` failed. The agent didn't
  change; the judge accepted it. A grounded-answer prompt that rejects a
  false claim without claiming ignorance is still worth writing.
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
  about two off-topic docs per question either way. Lowering `top_k` or
  adding a similarity threshold (the `/chat` log records each chunk's score)
  is the next retrieval change to measure.

**Before that** (5 docs, 65 cases, `router_v2` + `direct_answer_v2`,
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
Covered: agent routing (tool call → retrieval → grounded prompt, direct
answers, empty retrieval, tool call with no query argument), history reaching
both model calls, 429 retry/backoff, chunking, retrieval and ingestion, the
`/chat` endpoint and its log line, sessions (follow-ups, trimming, expiry,
per-visitor isolation), rate limits (each rule run against both the
in-memory and the Redis limiter via fakeredis, plus surviving a restart and
Redis outages), tracing and cost math, eval scoring and judge voting, and
golden-set integrity (unique ids, known categories, every expected doc id
exists in `data/docs/`, no doc answers an `unanswerable` case).

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

**Quota:** on the Gemini free tier, `gemini-3.5-flash-lite` (what
`gemini-flash-lite-latest` resolves to) allows 500 requests/day per Google
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
issue (`Retry-After: 60`) instead of a bare 500. A Gemini call that gets no
response within `GEMINI_TIMEOUT_S` (60s) is stopped rather than left hanging,
and `/chat` returns a 504 saying so. Retries only happen in a `/chat`'s
first 45s: a retry whose wait would end later isn't made, and the 429 or
503 above comes back straight away. Without that limit, Gemini's suggested
waits (up to ~60s each) added up across the nested retries: four
per-minute 429s in a row could keep a `/chat` going for 7 minutes, long
after the MCP proxy had given up on it. Now a `/chat` ends within 45s plus
60s for each of its three Gemini calls (route, embed, generate), 225s in
all, inside the MCP proxy's 240s. The eval harness calls the agent
directly, so it keeps retrying as before.

**Limits:** the quota is shared by everyone using the demo, so each visitor
(an IP address; for IPv6, its /64) can ask 6 questions a minute and 30 a
day. Past that, `/chat` returns a 429 with `Retry-After` and a `detail`
saying when to try again, without calling the model; the MCP tool passes
that message on. Change the numbers with `CHAT_LIMIT_PER_MINUTE` and
`CHAT_LIMIT_PER_DAY` (0 turns one off). With `REDIS_URL` set, counts (and
chat sessions) live in Redis. They survive restarts, which matters on
Render's free plan because the service restarts every time it wakes from
sleep: with counts in memory, each wake-up handed every visitor a fresh
allowance. The check and the count run as one Lua script, so simultaneous
requests can't slip past the limit. If Redis is unreachable, requests are
counted in memory instead, still limited, rather than all refused or all
let through. Without `REDIS_URL`, everything is in memory, and the
Dockerfile pins one worker process so the counts aren't split. A `query` can be up
to 2,000 characters and a request body up to 64 KB; a bigger body is
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
