# AI Customer Support Assistant — Backend

A FastAPI backend for an AI-driven customer support assistant that automates
responses across **chat** and **voice** channels, using a **Retrieval-Augmented
Generation (RAG)** pattern grounded in a knowledge base, with **conversation
memory** and **automatic escalation** to a human agent when confidence is low
or the user sounds frustrated.

## Architecture

```
                         ┌─────────────────────┐
                         │   Static test page   │  static/index.html
                         │  (HTML/CSS/fetch)     │
                         └──────────┬───────────┘
                                    │ POST /chat
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                          FastAPI (app/main.py)                     │
│                                                                     │
│  /chat ──┐                                                         │
│  /voice ─┤──► run_chat_pipeline()                                  │
│           │                                                        │
│           │  1. Heuristic frustration/human-request scan           │
│           │     (app/escalation.py)                                │
│           │  2. Load conversation history from MongoDB             │
│           │     (app/db.py — "conversation memory")                │
│           │  3. Embed query + retrieve top-k KB chunks             │
│           │     (app/vector_store.py — Chroma vector DB)           │
│           │  4. Call OpenAI chat completion, grounded in           │
│           │     retrieved context + history (app/llm.py)           │
│           │  5. Combine heuristic + model confidence/flag          │
│           │     → final escalate decision                          │
│           │  6. Persist both turns back to MongoDB                 │
│           └─────────────────────────────────────────────────────  │
└───────────────────────────────────────────────────────────────────┘
        │                          │                        │
        ▼                          ▼                        ▼
   MongoDB                   Chroma vector store        OpenAI API
 (conversations,             (embedded FAQ /            (chat, embeddings,
  escalations)                help-doc chunks)           Whisper, TTS)
```

### The RAG pattern, concretely

1. **Retrieve** — the incoming user message is embedded (locally, via Chroma's
   bundled MiniLM model by default — see "Choosing a chat provider" below)
   and compared against a Chroma vector index seeded from `app/knowledge_base/faq_docs.py`.
   The top-`k` most similar chunks (default `k=3`) are pulled back with their
   source titles and similarity scores.
2. **Augment** — those chunks are inserted into the system prompt as
   `KNOWLEDGE BASE CONTEXT`, alongside the last few turns of conversation
   history pulled from MongoDB, so the model has both *product knowledge* and
   *conversational context*.
3. **Generate** — the OpenAI chat model is instructed to answer **only** from
   that context, and to return structured JSON containing the reply plus a
   self-reported `confidence` score and an `escalate` flag, so the model
   participates in its own handoff decision instead of just producing free text.

### Conversation memory

Every turn (`user` and `assistant` message) is appended to a MongoDB document
keyed by `_id = conversation_id` (`app/db.py`). On each new request, the last
`MAX_HISTORY_TURNS` turns are pulled back and fed into the prompt — this is
what lets the assistant answer follow-ups like *"and what happens if I go over
that limit?"* without the user repeating themselves.

### Escalation logic

Two independent signals are combined (`app/escalation.py`):

- **Heuristic (pre-generation):** regex-based scan of the raw user message for
  frustrated tone ("this is ridiculous", excessive `!!`), explicit requests
  for a human ("talk to a person", "escalate", "manager"), and repeated-question
  language ("I already asked", "for the third time").
- **Model self-assessment (post-generation):** the LLM returns its own
  `confidence` (0–1) and `escalate` boolean based on whether the retrieved
  context was sufficient to answer correctly.

If **either** the heuristic fires, the model flags it, or confidence falls
below `ESCALATION_CONFIDENCE_THRESHOLD` (default `0.55`), the conversation is
marked `escalated: true` and logged to the `escalations` collection, retrievable
via `GET /escalations` — this is the queue a human-handoff dashboard would poll.

### Voice channel

`POST /voice` accepts an audio file, transcribes it with the OpenAI Whisper
API (`app/llm.py::transcribe_audio`), and feeds the transcript through the
**exact same** `run_chat_pipeline()` used by `/chat` — so voice and chat share
identical retrieval, memory, and escalation behavior. Optionally
(`respond_with_audio=true`), the reply is also synthesized to speech via the
OpenAI TTS API and served back at `/voice/audio/{filename}`.

## Project layout

```
app/
  main.py              FastAPI app, /chat, /voice, /conversations, /escalations
  config.py            Environment-driven settings
  db.py                MongoDB (motor async client) + conversation memory helpers
  models.py            Pydantic request/response schemas
  llm.py               OpenAI chat/embeddings/Whisper/TTS wrapper
  vector_store.py      Chroma vector store: chunking, upsert, retrieval
  escalation.py         Heuristic frustration/handoff detection
  seed_kb.py           One-off script to embed & load the FAQ knowledge base
  knowledge_base/
    faq_docs.py        Example FAQ / help-doc source content
static/
  index.html           Minimal HTML/CSS/fetch test console for /chat
postman/
  customer_support_ai.postman_collection.json
requirements.txt
docker-compose.yml     Local MongoDB for development
.env.example
```

## Running locally

### 1. Prerequisites

- Python 3.10+
- Docker (for local MongoDB), or a MongoDB instance you already have
- An OpenAI API key

### 2. Setup

```bash
git clone <this-repo>
cd customer-support-ai

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env -- see "Choosing a chat provider" below
```

#### Choosing a chat provider: OpenAI vs. OpenRouter (free)

This project talks to any OpenAI-*compatible* chat API, so you can use either:

**OpenAI directly** (paid, needs billing set up on your account):
```
CHAT_API_KEY=sk-proj-your-real-openai-key
CHAT_BASE_URL=
CHAT_MODEL=gpt-4o-mini
```

**OpenRouter** (free tier, no credit card required):
1. Sign up at https://openrouter.ai and create an API key on the Keys page.
2. Browse https://openrouter.ai/models?max_price=0 for a currently-free model
   — the free lineup rotates week to week, so check the live list rather than
   trusting a hardcoded example. Free model IDs always end in `:free`.
3. Set in `.env`:
   ```
   CHAT_API_KEY=sk-or-v1-your-openrouter-key
   CHAT_BASE_URL=https://openrouter.ai/api/v1
   CHAT_MODEL=<provider>/<model>:free
   ```

Either way, **knowledge-base retrieval works the same and needs no API key at
all** by default — embeddings run locally via a small model bundled with
Chroma (`USE_LOCAL_EMBEDDINGS=true` in `.env.example`), downloaded once on
first run from a public CDN and cached locally after that.

The only feature that strictly requires a real, billed **OpenAI** key is the
optional `/voice` endpoint (Whisper transcription + TTS) — OpenRouter doesn't
proxy audio endpoints. Leave `OPENAI_API_KEY` blank in `.env` to skip voice;
`/chat` and the test console work fully without it, and `/voice` will return
a clear `501` explaining why instead of crashing.

### 3. Start MongoDB

```bash
docker compose up -d
```

(Or point `MONGODB_URI` in `.env` at an existing MongoDB deployment, e.g. Atlas.)

### 4. Seed the knowledge base

This embeds the example FAQ docs and stores them in a local Chroma index
under `./chroma_data`:

```bash
python -m app.seed_kb
```

### 5. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive API docs (Swagger UI): **http://localhost:8000/docs**
- Test console (plain HTML page): **http://localhost:8000/**
- Health check: **http://localhost:8000/health**

### 6. Try it

Via the test page, or with curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-1", "message": "How much storage does the Pro plan include?"}'
```

Follow up in the same conversation:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-1", "message": "What happens if I go over that?"}'
```

Trigger escalation:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "demo-2", "message": "This is ridiculous, let me talk to a real person!!"}'
```

Voice:

```bash
curl -X POST "http://localhost:8000/voice?respond_with_audio=true" \
  -F "audio=@/path/to/question.wav"
```

A ready-made Postman collection covering all of the above is at
`postman/customer_support_ai.postman_collection.json` — import it and set the
`base_url` collection variable if you're not running on `localhost:8000`.

## Deploying to Render

Render works well for this project, with two adjustments from local dev:
Render has **no managed MongoDB** (only Postgres and Key Value/Redis), and
its **free-tier disk is ephemeral** — wiped on every deploy/restart. Both are
already handled in this repo:

- `app/main.py` **auto-seeds the knowledge base on startup** if the vector
  store is empty, so an ephemeral disk is a non-issue for this small example
  KB — no manual `seed_kb` step needed after each deploy.
- `render.yaml` (a Render Blueprint) declares the service and every env var
  this app reads, so Render can set most of it up for you.

### 1. Get a free MongoDB Atlas cluster (replaces your local `docker compose` Mongo)

1. Sign up at https://www.mongodb.com/cloud/atlas → create a free **M0** cluster.
2. Database Access → add a database user (username + password).
3. Network Access → add `0.0.0.0/0` (allow from anywhere) — simplest for a
   demo; tighten to Render's static outbound IPs for production.
4. Get your connection string (Connect → Drivers): looks like
   `mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net`.

### 2. Deploy the Blueprint

1. Push this repo to GitHub.
2. In Render: **New +** → **Blueprint** → select the repo → it reads `render.yaml`.
3. Render will prompt for the vars marked `sync: false` (secrets it won't
   store in the YAML): `CHAT_API_KEY`, `MONGODB_URI`, and optionally
   `OPENAI_API_KEY` if you want `/voice`. Paste your OpenRouter key and Atlas
   connection string in there.
4. Deploy. After the first successful deploy, update the
   `OPENROUTER_SITE_URL` env var to your actual `https://<service>.onrender.com`
   URL (cosmetic — only used in OpenRouter's attribution headers) and redeploy.

No Blueprint? Create a **Web Service** manually instead, pointed at this repo:
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Add the same env vars listed in `render.yaml` through the dashboard's
  Environment tab.

### What to expect on the free tier

- **Cold starts**: free services spin down after 15 minutes idle and take
  roughly a minute to wake on the next request — the first request after a
  quiet period will be slow (Mongo reconnect + a fresh knowledge-base
  auto-seed, which also re-downloads Chroma's small local embedding model
  since the disk was wiped). Subsequent requests are fast.
- **Voice stays optional**: leave `OPENAI_API_KEY` blank in Render and
  `/voice` returns a clean `501` instead of failing oddly; `/chat` is
  unaffected.
- **OpenRouter free-model rotation**: same caveat as local dev —
  `CHAT_MODEL=openrouter/free` auto-routes to whatever's currently free, so
  you don't need to redeploy every time a specific `:free` model gets pulled.

## Extending this

- **Bigger knowledge base:** replace `app/knowledge_base/faq_docs.py` with a
  loader that pulls from your real help center / Confluence / Zendesk export,
  and re-run `python -m app.seed_kb`. For large corpora, swap Chroma for a
  managed vector DB (Pinecone, Qdrant, pgvector) — `app/vector_store.py` is the
  only file that needs to change.
- **Smarter escalation:** swap the regex heuristic in `app/escalation.py` for
  a fine-tuned sentiment/intent classifier; the combination logic in
  `combine_escalation_signals()` already treats it as one signal among several.
- **Human handoff integration:** have `db.log_escalation()` also push to a
  ticketing system (Zendesk, Intercom, Slack) via a webhook.
- **Auth:** endpoints are currently open for local testing — add an API key
  or OAuth dependency in `main.py` before deploying.