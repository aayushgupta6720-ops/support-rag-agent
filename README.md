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
  core/config.py       Settings, loaded from .env
  core/gemini_client.py Shared, cached google-genai client
  models/schemas.py    Pydantic request/response models
  rag/
    chunking.py        Paragraph-packing text chunker
    embeddings.py       Gemini embedding calls (gemini-embedding-001)
    qdrant_store.py     Async Qdrant client, collection setup, search
    ingest.py           Chunk + embed + upsert documents
    retrieval.py        Embed a query and fetch top-k chunks
  agent/
    graph.py           LangGraph agent: route -> (retrieve) -> generate
    prompts.py         Versioned system prompts
    tools.py           search_support_docs tool declaration
  eval/
    dataset.py         Loads data/eval/golden_set.jsonl
    judge.py           LLM-as-judge: grades an answer against a reference
    harness.py         Runs the agent + judge over the golden set
    prompts.py         Versioned judge prompt
  core/
    observability.py  Per-call trace (latency/tokens/cost per step) + JSON logging
    pricing.py         Approximate per-token/per-char cost estimates
scripts/
  ingest.py            CLI: loads data/docs/*.md and ingests into Qdrant
  eval.py              CLI: runs the eval harness, prints + saves a report
data/docs/             Sample support docs used by the ingestion script
data/eval/             Golden set + timestamped eval run results
mcp_server/            Optional MCP wrapper (separate venv, see below)
Dockerfile
requirements.txt
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

Then `POST /chat` with `{"query": "..."}`. A LangGraph agent routes the query:
a router call decides whether to call the `search_support_docs` tool or
answer directly (e.g. for greetings), then a generation call produces a
structured, schema-validated answer grounded in whatever was retrieved.

Run the eval harness (agent + LLM-as-judge over a golden set) with:

```bash
python -m scripts.eval
```

It reports a judge pass rate and retrieval hit rate, and saves a timestamped
JSON report under `data/eval/results/`.

Every `/chat` call emits one structured JSON log line (stdout) with a
per-step latency/token/cost breakdown (routing, embedding, vector search,
generation), the routing decision, retrieval scores, and which prompt
versions were used — e.g.:

```json
{"event": "chat_call", "query": "...", "used_tool": true, "num_chunks_retrieved": 4,
 "retrieval_scores": [0.72, 0.58], "router_prompt_version": "router_v1",
 "answer_prompt_version": "grounded_answer_v1", "latency_ms": 3427.81,
 "total_tokens": 956, "total_cost_usd": 8.7e-05,
 "steps": [{"name": "route", "latency_ms": 1358.47, ...}, ...]}
```

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

## Optional: MCP wrapper

`mcp_server/` exposes `/chat` as an MCP tool (`ask_support_agent`) so an MCP
client (e.g. Claude Desktop) can call the agent directly. It's a thin HTTP
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
