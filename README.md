# ParcelPilot Support Agent

AI-powered support agent for ParcelPilot's B2B logistics platform, built as a submission for the CalQuity AI Engineer assessment.

**Stack:** Mistral Large 2 · LangGraph · FastAPI + WebSocket · HuggingFace Inference API (BAAI/bge-small-en-v1.5) · Supabase (pgvector + structured data)

---

## What Was Built

Both user contexts from the assessment spec are implemented in a single application:

- **Customer Portal** - customers authenticate with account ID + PIN and can query their own account data, SLA terms, open tickets, order status, and policy documents. Hard-scoped to their account at the data layer.
- **Internal Ops** - authorised ParcelPilot staff authenticate with username + password and get a proactive issue scan on login, cross-account data access, SLA breach detection, policy conflict flagging, and the ability to create escalation tickets with a confirmation gate.

---

## Minimum Requirements Coverage

| Requirement | Implementation |
|---|---|
| Natural-language chatbot | LangGraph agent with Mistral Large 2; streaming responses over WebSocket |
| Access control | Supabase RLS + Python WHERE injection on every query; model cannot bypass |
| Document search tool | `search_documents` - HF embed + pgvector semantic search |
| Structured data tool | `query_data` - account, order, ticket lookups scoped by role |
| State-changing action tool | `create_action` - creates escalation tickets; mocked locally, logs to `actions_log` |
| Confirmation gate | `execute=False` draft → `PENDING_CONFIRMATION` state → user must reply yes/no |
| Multi-step requests | Agent chains tools freely: e.g. order lookup → account fetch → agreement search → SLA calculation → escalation decision |
| Chat interface | Custom FastAPI + WebSocket UI served at `http://localhost:8000`; shows source chips and active tool state |

---

## Architecture Note

### Agent Design

```
ui/static/index.html  (custom chat UI - vanilla JS + WebSocket)
    │  WebSocket ws://localhost:8000/ws/{session_id}
    ▼
ui/server.py  (FastAPI + WebSocket server)
    │
    ▼
agent/graph.py  (LangGraph StateGraph)
    ├── agent_node     - Mistral Large 2 with function calling
    ├── tool_node      - dispatches to 3 tools, enforces access scope
    └── confirm_node   - intercepts create_action(execute=False) and pauses for yes/no
```

The graph runs `agent_node → tool_node` in a loop until the model stops calling tools, then streams the final response. If a tool call produces a `PENDING_CONFIRMATION` state, the graph pauses and waits for the next user message before executing.

### Tool Design

**`search_documents`** - takes a query string, embeds it via HuggingFace `BAAI/bge-small-en-v1.5`, runs a cosine similarity search against `document_chunks` in Supabase pgvector, and returns the top-K chunks with their source document name and authority tier. The tool returns source metadata alongside content so the agent can cite authority.

**`query_data`** - accepts a structured intent (accounts, orders, tickets) and filters. For customer sessions, the Supabase client is initialised with `set_session_context(account_id)` before every query, and RLS policies reject any row not belonging to that account. For internal sessions, the same queries run without the account filter.

**`create_action`** - accepts a ticket/escalation payload and an `execute` flag. When `execute=False`, it returns a draft for the user to review. When `execute=True` (after confirmation), it inserts into `actions_log`. Mocked locally - no external ticketing system integration.

### Document and Structured-Data Handling

PDFs are chunked at 600 tokens with 100-token overlap (preserving sentence boundaries) and embedded at ingestion time. Each chunk stores its source document name. The xlsx is parsed into three Supabase tables: `accounts`, `orders`, `tickets`.

The system prompt tells the model to prefer structured data for factual lookups (order status, ticket state, account plan) and document search for policy/SLA/agreement questions. It also tells the model to use `query_data` first for account-specific questions since structured data is more authoritative than policy text for instance-level facts.

### Source Reliability and Conflict Handling

Source authority is tiered and enforced in the system prompt and in the UI:

1. **Customer Agreement** (Northstar, LumenWorks) - highest authority; overrides all general policy
2. **Support Policy v3 CURRENT** - current general policy
3. **SOPs** (Cancellation and Credit, Product Ops) - operational detail
4. **Support Policy v2 DEPRECATED** - lowest; used only when v3 has no coverage
5. **Historical ticket resolutions** - context only; may be incorrect

When the agent retrieves chunks from both a customer agreement and a general policy on the same topic, it is instructed to apply the agreement and flag the conflict. The UI surfaces a "Source conflict - higher-authority document applied" chip on any response where this occurred. This chip is injected by the server when it detects multi-source responses with differing authority tiers.

### Major Technical Trade-offs

**Mistral Large 2 over a smaller model** - the assessment data requires multi-step tool chaining (order + account + agreement + SLA calculation in one query). Smaller models dropped tool calls mid-chain. The rate limit (1 req/s on free tier) is the cost; mitigated by streaming and per-session WebSocket isolation.

**FastAPI + WebSocket over Chainlit** - Chainlit's session model made it hard to implement the dual auth flow (customer PIN vs staff password) within a single app without routing hacks. A custom UI gives full control over the auth gate, mode switching, source chip rendering, and the confirmation gate UX.

**Supabase RLS as the enforcement layer** - model-instruction-only access control fails under adversarial prompts. RLS makes the enforcement cryptographic: a query issued under the wrong session context returns zero rows regardless of what the model asks for.

**HuggingFace Inference API for embeddings** - avoids a local GPU requirement. The cold-start latency (first embed call after model sleep is ~3s) is the trade-off; acceptable for a demo but would need a dedicated endpoint in production.

---

## Additional Client Problems Addressed

### Problem 1: Proactive Issue Detection

Implemented as an automatic scan that fires when internal ops staff log in. The scanner (`proactive/detector.py`) queries open tickets in real time and checks for:

- SLA breach risk - tickets where elapsed time exceeds the applicable SLA target (resolved from customer agreement if present, else policy v3 defaults)
- Stuck or failed pickups - orders where carrier status is stale relative to expected pickup window
- Ticket clusters - groups of 2+ tickets with similar subjects across one or more accounts (simple keyword clustering; flags potential product-wide issues)

Results are pushed as a system message before the first user turn so staff immediately see what needs attention without asking.

### Problem 2: Trust and Reliability

Three concrete mechanisms:

1. **Source authority hierarchy** - described above; enforced in prompt and surfaced in UI chips.
2. **Conflict detection** - when retrieval returns chunks from documents of different authority tiers covering the same topic, the server detects the overlap and injects a conflict flag into the response context. The agent is instructed to state which source it applied and why.
3. **Confirmation gate** - no state-changing action executes without explicit user yes/no. The draft is shown before execution so staff can review the exact ticket content before it is created.

---

## What I Would Build Next (Prioritised)

1. **Hosted deployment** - the assessment prefers a hosted link. The app is ready to deploy on Render or Railway (FastAPI + static files); blocked only by Supabase connection string exposure in env. Would use secrets management and deploy within a day.

2. **Feedback loop on source conflicts** - when staff override a conflict decision ("use the policy, not the agreement"), log the override. Feed overrides back as fine-tuning signal or few-shot examples in the prompt to reduce future false conflicts.

3. **Real ticket system integration** - `create_action` currently writes to `actions_log`. Connecting it to Linear, Zendesk, or Freshdesk would make escalations actionable without a human copy-paste step. High value for the ops team; low engineering cost given the tool interface is already abstracted.

4. **Per-agent memory** - currently each conversation starts fresh. Storing a short summary of each resolved ticket (account, issue, resolution) in a vector table would let the agent answer "has this customer had this issue before?" without re-reading the full ticket history.

5. **Eval harness** - the 19 functional test cases in `tests/functional_tests.py` cover the happy path. A continuous eval harness that runs on every deploy against a fixed set of adversarial prompts (prompt injection, cross-account data requests, deprecated policy citations) would catch regressions before they reach staff.

**What I intentionally left out:**
- Real email/notification delivery on escalation (mocked locally as intended)
- Multi-language support (all source docs are English)
- Fine-tuning (Mistral Large 2 zero-shot is accurate enough for this data volume; fine-tuning would matter at 10x the ticket volume)

**One metric to judge usefulness:** first-contact resolution rate - the percentage of customer queries answered fully by the agent without a human follow-up. Baseline from the current manual process is the denominator; anything above 60% first-contact on policy/SLA questions would justify the ops time saved.

---

## AI Tool Usage

Built end-to-end with **Claude Code** (claude-sonnet-4-6) used for:
- Architecture design and LangGraph graph structure
- Writing and debugging the FastAPI WebSocket server
- Supabase schema, RLS policy SQL, and vector search function
- System prompt design and source authority logic
- Functional test suite (19 test cases run both via script and manual browser)
- README and documentation

All code was reviewed, tested, and run locally before commit. The agent design decisions (authority hierarchy, conflict detection approach, confirmation gate implementation) were made by me; Claude implemented them.

---

## Setup

### Prerequisites

- Python 3.11+
- A [Supabase](https://app.supabase.com) project with pgvector enabled
- [Mistral AI](https://console.mistral.ai) API key
- [HuggingFace](https://huggingface.co/settings/tokens) API token (free tier works)

### 1. Clone and install

```bash
git clone https://github.com/SanskarBajaj123/ParcelPilot.git
cd ParcelPilot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

```
MISTRAL_API_KEY=your_mistral_key_here
HF_TOKEN=your_huggingface_token_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

> **Supabase keys:** Project Settings - API. Anon key for the chatbot; service role key for ingestion only.

### 3. Set up Supabase schema

Run in [Supabase Dashboard - SQL Editor](https://app.supabase.com):

```
scripts/setup_supabase.sql
```

Creates: `accounts`, `orders`, `tickets`, `document_chunks` (pgvector), `actions_log`, all RLS policies, and the vector search function.

### 4. Add data files

```
data/
  docs/
    01_Support_Policy_v3_CURRENT.pdf
    02_Support_Policy_v2_DEPRECATED.pdf
    03_Cancellation_and_Service_Credit_SOP_v4.pdf
    04_Product_Operations_Guide_and_Known_Issues.pdf
    05_Northstar_Logistics_Enterprise_Agreement.pdf
    06_LumenWorks_Service_Agreement.pdf
    parcelpilot_data.xlsx
```

### 5. Run ingestion

```bash
python scripts/ingest.py
```

Parses PDFs, embeds chunks via HuggingFace, upserts to Supabase. Takes 2-5 minutes.

### 6. Launch

```bash
python ui/server.py
```

Open [http://localhost:8000](http://localhost:8000).

---

## Architecture Diagram

```
ui/static/index.html  (chat UI)
    │  WebSocket
    ▼
ui/server.py  (FastAPI)
    │
    ▼
agent/graph.py  (LangGraph StateGraph)
    ├── agent_node     - Mistral Large 2 (function calling)
    ├── tool_node      - 3 tools, role-scoped
    └── confirm_node   - confirmation gate for actions

Tools:
  agent/tools/search_documents.py  - HF embed → pgvector
  agent/tools/query_data.py        - Supabase structured queries (RLS enforced)
  agent/tools/create_action.py     - draft/execute pattern → actions_log

Supporting:
  agent/state.py      - AgentState TypedDict
  agent/history.py    - time-based message filtering (1h window)
  agent/prompts.py    - system prompt with source authority hierarchy
  proactive/detector.py  - SLA breach + failed-pickup + cluster scanner

Infrastructure:
  scripts/setup_supabase.sql  - schema + RLS + vector search function
  scripts/ingest.py           - PDF + xlsx ingestion pipeline
  tests/functional_tests.py   - 19 functional test cases
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MISTRAL_API_KEY` | Yes | - | Mistral AI API key |
| `HF_TOKEN` | Yes | - | HuggingFace Inference API token |
| `SUPABASE_URL` | Yes | - | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | - | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | - | Service role key (ingestion only) |
| `HISTORY_WINDOW_HOURS` | No | `1` | Conversation history window |
| `MAX_HISTORY_MESSAGES` | No | `30` | Hard cap on message count |
| `RETRIEVAL_TOP_K` | No | `5` | Document chunks per search |
