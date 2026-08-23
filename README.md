# ParcelPilot Support Agent

AI-powered support agent for ParcelPilot's B2B logistics platform, built as a submission for the CalQuity AI Engineer assessment.

**Stack:** Mistral Large 2 - LangGraph - FastAPI + WebSocket - HuggingFace Inference API (BAAI/bge-small-en-v1.5) - Supabase (pgvector + structured data + conversation history)

**Hosted:** https://parcelpilot-agent.onrender.com

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
| Confirmation gate | Natural conversation flow: agent proposes with `execute=False`, asks "Shall I go ahead?", reads DB history on next turn to detect user's yes/no before calling `execute=True` |
| Multi-step requests | Agent chains tools freely: e.g. order lookup - account fetch - agreement search - SLA calculation - escalation decision |
| Chat interface | Custom FastAPI + WebSocket UI served at `http://localhost:8000`; streams tokens in real time and shows active tool calls as they happen |

---

## Architecture

### Agent Design

```
ui/static/index.html  (custom chat UI - vanilla JS + WebSocket)
    |  WebSocket ws://localhost:8000/ws
    v
ui/server.py  (FastAPI + WebSocket server)
    |  saves user message to Supabase conversation_messages
    |  fetches last 3 user + 3 assistant messages from DB
    |  builds fresh AgentState with DB history each turn
    v
agent/graph.py  (LangGraph StateGraph)
    |
    START -> agent_node
               |
               +-- (tool calls?) --> tool_node --> agent_node (loop)
               |
               +-- (no tool calls) --> END
```

The graph is intentionally simple: two nodes in a loop. Confirmation is handled entirely through natural conversation - the agent proposes an action, asks "Shall I go ahead?", and reads the DB-persisted history on the next turn to detect the user's reply.

### Confirmation Flow

This is the most important design decision in the codebase:

```
Turn 1 (user asks for an action):
  agent_node calls create_action(execute=False)
  tool_node returns draft summary
  agent_node generates: "Here's what I'll do: [draft]. Shall I go ahead?"
  ui/server.py saves this as assistant message to conversation_messages

Turn 2 (user replies "yes"):
  ui/server.py saves "yes" as user message
  ui/server.py fetches last 3+3 messages from DB (includes Turn 1 exchange)
  Fresh AgentState is built with that history
  agent_node sees the history: prior proposal + user's "yes"
  agent_node calls create_action(execute=True)
  Action is committed to actions_log
```

No state machine, no in-memory session state, no pending_action flag. The LLM reads the conversation history and makes the decision, just like a human would.

### Tool Design

**`search_documents`** - takes a query string, embeds it via HuggingFace `BAAI/bge-small-en-v1.5`, runs a cosine similarity search against `document_chunks` in Supabase pgvector, and returns the top-K chunks with their source document name and authority tier. The tool returns source metadata alongside content so the agent can cite authority.

**`query_data`** - accepts a structured intent (accounts, orders, tickets) and filters. For customer sessions, the Supabase client is initialised with `set_session_context(account_id)` before every query, and RLS policies reject any row not belonging to that account. For internal sessions, the same queries run without the account filter. Intents: `lookup_order`, `lookup_account`, `list_open_tickets`, `calculate_credit_eligibility`, `check_sla_breach`.

**`create_action`** - accepts a ticket/escalation payload and an `execute` flag. When `execute=False`, it returns a draft for the user to review (no side effects). When `execute=True` (after confirmation via conversation history), it inserts into `actions_log`. Mocked locally - no external ticketing system integration. Action types: `escalate_ticket`, `update_ticket_status`, `create_followup_task`.

### Preemptive Document Retrieval

On every user turn (before the LLM responds), `agent_node` runs a preemptive document search against the user's query. For customer sessions, it also runs a targeted search for their specific agreement (e.g. Northstar Enterprise Agreement) to ensure it is always in context. This prevents the model from answering from memory when a policy document exists.

### Conversation History (DB-backed)

Each turn:
1. User message is saved to `conversation_messages` table (session_id, user_id, role, content)
2. Last 3 user + last 3 assistant messages are fetched, sorted chronologically
3. A fresh `AgentState` is built with those messages as history
4. After streaming, assistant reply is saved to `conversation_messages`

This means the LLM always has the prior exchange in context, enabling the natural confirmation flow without any in-memory state.

### Document and Structured-Data Handling

PDFs are chunked at 600 tokens with 100-token overlap (preserving sentence boundaries) and embedded at ingestion time. Each chunk stores its source document name. The xlsx is parsed into three Supabase tables: `accounts`, `orders`, `tickets`.

The system prompt tells the model to prefer structured data for factual lookups (order status, ticket state, account plan) and document search for policy/SLA/agreement questions.

### Source Authority Hierarchy

Source authority is tiered and enforced in the system prompt and in the UI:

1. **Customer Agreement** (Northstar, LumenWorks) - highest authority; overrides all general policy
2. **Support Policy v3 CURRENT** (effective 1 May 2026) - current general policy
3. **SOPs** (Cancellation and Credit, Product Ops) - operational detail
4. **Historical ticket resolutions** - context only; may be incorrect

**Deprecated source (02_Support_Policy_v2):** explicitly excluded - if retrieved, the agent ignores it.

When the agent retrieves chunks from both a customer agreement and a general policy on the same topic, it applies the agreement and flags the conflict in its response - stating which source governed and why the lower-authority source was overridden.

### Major Technical Trade-offs

**Mistral Large 2 over a smaller model** - the assessment data requires multi-step tool chaining (order + account + agreement + SLA calculation in one query). Smaller models dropped tool calls mid-chain. The rate limit (1 req/s on free tier) is the cost; mitigated by streaming and per-session WebSocket isolation.

**FastAPI + WebSocket over Chainlit** - Chainlit's session model made it hard to implement the dual auth flow (customer PIN vs staff password) within a single app without routing hacks. A custom UI gives full control over the auth gate, mode switching, source chip rendering, and the confirmation gate UX.

**Supabase RLS as the enforcement layer** - model-instruction-only access control fails under adversarial prompts. RLS makes the enforcement structural: a query issued under the wrong session context returns zero rows regardless of what the model asks for.

**HuggingFace Inference API for embeddings** - avoids a local GPU requirement. The cold-start latency (first embed call after model sleep is ~3s) is the trade-off; acceptable for a demo but would need a dedicated endpoint in production.

**DB-backed history over in-memory session state** - enables the natural confirmation flow and means conversation history survives server restarts. The trade-off is a Supabase round-trip per turn, which adds ~50ms.

---

## Additional Client Problems Addressed

### Problem 1: Proactive Issue Detection

Implemented as an automatic scan that fires when internal ops staff log in. The scanner (`proactive/detector.py`) queries open tickets in real time and checks for:

- **SLA breach risk** - tickets where elapsed time exceeds the applicable SLA target (resolved from customer agreement if present, else policy v3 defaults)
- **Stuck or failed pickups** - orders where carrier status is stale relative to expected pickup window
- **Ticket clusters** - groups of 2+ tickets with similar subjects across one or more accounts (Jaccard similarity on keyword sets; flags potential product-wide issues)
- **Multi-customer patterns** - clusters that span 2+ accounts are flagged separately as potential platform-wide issues

Results are pushed as a system message before the first user turn so staff immediately see what needs attention without asking.

### Problem 2: Trust and Reliability

Three concrete mechanisms:

1. **Source authority hierarchy** - described above; enforced in prompt and surfaced in UI chips.
2. **Conflict detection** - when retrieval returns chunks from documents of different authority tiers covering the same topic, the server detects the overlap and injects a conflict flag. The agent states which source it applied and why.
3. **Confirmation gate** - no state-changing action executes without explicit user yes/no via natural conversation. The draft is shown before execution so staff can review exact content before it is committed.

---

## What I Would Build Next (Prioritised)

1. **Real ticket system integration** - `create_action` currently writes to `actions_log`. Connecting it to Linear, Zendesk, or Freshdesk would make escalations actionable without a human copy-paste step.

2. **Feedback loop on source conflicts** - when staff override a conflict decision, log the override. Feed overrides back as fine-tuning signal or few-shot examples to reduce future false conflicts.

3. **Per-agent memory** - currently each conversation starts fresh (last 6 messages). Storing a short summary of each resolved ticket in a vector table would let the agent answer "has this customer had this issue before?" without re-reading the full ticket history.

4. **Eval harness** - the functional tests cover the happy path. A continuous eval harness that runs on every deploy against adversarial prompts (prompt injection, cross-account data requests, deprecated policy citations) would catch regressions before they reach staff.

**What I intentionally left out:**
- Real email/notification delivery on escalation (mocked locally as intended)
- Multi-language support (all source docs are English)
- Fine-tuning (Mistral Large 2 zero-shot is accurate enough for this data volume)

**One metric to judge usefulness:** first-contact resolution rate - the percentage of customer queries answered fully by the agent without a human follow-up. Anything above 60% first-contact on policy/SLA questions would justify the ops time saved.

---

## AI Tool Usage

Built end-to-end with **Claude Code** (claude-sonnet-4-6) used for:
- Architecture design and LangGraph graph structure
- Writing and debugging the FastAPI WebSocket server
- Supabase schema, RLS policy SQL, and vector search function
- System prompt design and source authority logic
- Functional test suite
- README and documentation

All code was reviewed, tested, and run locally before commit. The agent design decisions (authority hierarchy, conflict detection approach, DB-backed confirmation flow) were made by me; Claude implemented them.

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

Creates: `accounts`, `orders`, `tickets`, `document_chunks` (pgvector), `conversation_messages` (chat history), `actions_log`, all RLS policies, and the vector search function.

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
    |  WebSocket
    v
ui/server.py  (FastAPI)
    |  saves/fetches conversation_messages (Supabase)
    v
agent/graph.py  (LangGraph StateGraph)
    +-- agent_node  - Mistral Large 2 (function calling + preemptive search)
    +-- tool_node   - 3 tools, role-scoped, account-isolated

Tools:
  agent/tools/search_documents.py  - HF embed -> pgvector cosine similarity
  agent/tools/query_data.py        - Supabase structured queries (RLS enforced)
  agent/tools/create_action.py     - draft/execute pattern -> actions_log

Supporting:
  agent/state.py      - AgentState TypedDict (messages, user_context, sources_used)
  agent/history.py    - message filtering
  agent/prompts.py    - system prompt with source authority hierarchy + confirmation steps
  proactive/detector.py  - SLA breach + failed-pickup + Jaccard cluster scanner

Infrastructure:
  scripts/setup_supabase.sql  - schema + RLS + pgvector search function
  scripts/ingest.py           - PDF + xlsx ingestion pipeline
  tests/functional_tests.py   - functional test cases
```

---

## Test Credentials

### Customer Portal

| Account | PIN | Name |
|---|---|---|
| ACCT-001 | 1234 | Northstar Logistics |
| ACCT-002 | 5678 | LumenWorks |
| ACCT-003 | 4321 | (demo account) |
| ACCT-004 | 8765 | (demo account) |

### Internal Ops

| Username | Password |
|---|---|
| priya | ParcelPilot@2026 |
| rohit | Support@2026! |
| maya | Ops#Secure99 |
| admin | Admin$Master1 |

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MISTRAL_API_KEY` | Yes | - | Mistral AI API key |
| `HF_TOKEN` | Yes | - | HuggingFace Inference API token |
| `SUPABASE_URL` | Yes | - | Supabase project URL |
| `SUPABASE_ANON_KEY` | Yes | - | Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | - | Service role key (ingestion + actions) |
| `MAX_HISTORY_TURNS` | No | `3` | Number of prior turns to feed LLM per message |
