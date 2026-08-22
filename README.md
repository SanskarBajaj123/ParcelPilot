# ParcelPilot Support Agent

AI-powered support chatbot for ParcelPilot's B2B logistics platform.

**Stack:** Mistral Large 2 · LangGraph · HuggingFace Inference API (BAAI/bge-small-en-v1.5) · Supabase (pgvector + structured data) · Chainlit

---

## Features

| Feature | Detail |
|---|---|
| Dual modes | Customer-facing (scoped to own account) + Internal Ops (full access) |
| RAG | Semantic search over policies, SOPs, customer agreements (pgvector) |
| Source authority | Customer Agreement > Policy v3 > SOPs > Historical tickets |
| Conflict detection | Python-layer detection; injected into context before LLM |
| Tool use | `search_documents`, `query_data`, `create_action` |
| Confirmation gate | State-changing actions require explicit user yes/no |
| Access control | Supabase RLS + Python WHERE injection; model cannot bypass |
| Time-based history | Configurable window (default 1h), tool-pair repair |
| Proactive scan | Internal ops: SLA breach + failed-pickup detection on startup |

---

## Prerequisites

- Python 3.11+
- A [Supabase](https://app.supabase.com) project with pgvector enabled
- [Mistral AI](https://console.mistral.ai) API key
- [HuggingFace](https://huggingface.co/settings/tokens) API token (free tier is fine)

---

## Setup

### 1. Clone and install

```bash
git clone <YOUR_REPO_URL>
cd parcelpilot-agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```
MISTRAL_API_KEY=your_mistral_key_here
HF_TOKEN=your_huggingface_token_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

> **Supabase keys:** Go to Project Settings → API. The anon key is for the chatbot; the service role key is for ingestion only (bypasses RLS).

### 3. Set up Supabase schema

In the [Supabase Dashboard → SQL Editor](https://app.supabase.com), open and run:

```
scripts/setup_supabase.sql
```

This creates: `accounts`, `orders`, `tickets`, `document_chunks` (pgvector), `actions_log`, all RLS policies, and the vector search function.

### 4. Add data files

Copy the ParcelPilot data pack into the `data/docs/` directory:

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

This will:
- Parse all PDFs → chunk → embed via HuggingFace → insert into `document_chunks`
- Read the xlsx → upsert `accounts`, `orders`, `tickets`

Ingestion takes ~2–5 minutes depending on HuggingFace model load time.

### 6. Launch the chatbot

```bash
chainlit run ui/app.py --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Architecture

```
ui/app.py  (Chainlit)
    │
    ▼
agent/graph.py  (LangGraph StateGraph)
    ├── agent_node     ← Mistral Large 2 with function calling
    ├── tool_node      ← dispatches to 3 tools
    └── confirm_node   ← yes/no gate for state-changing actions

Tools:
  agent/tools/search_documents.py  ← HF embed → Supabase pgvector
  agent/tools/query_data.py        ← Supabase structured queries
  agent/tools/create_action.py     ← draft / execute pattern

Supporting:
  agent/state.py      ← AgentState TypedDict
  agent/history.py    ← time-based message filtering (1h window)
  agent/prompts.py    ← 7-block system prompt + retrieved context builder
  proactive/detector.py  ← SLA breach + failed-pickup scanner (internal)

Infrastructure:
  scripts/setup_supabase.sql  ← full schema + RLS + search function
  scripts/ingest.py           ← PDF + xlsx → Supabase
```

---

## Conversation History

History is filtered to the last **1 hour** (configurable via `HISTORY_WINDOW_HOURS` in `.env`). Tool-call / ToolMessage pairs are never split. A minimum of 4 messages is always retained regardless of age.

---

## Access Control

- **Customer mode:** RLS policies on `accounts`, `orders`, `tickets` restrict all queries to `account_id = current_setting('app.account_id')`. Set before every query via the `set_session_context` SQL function.
- **Internal mode:** Service role key is used only for ingestion. Chatbot uses anon key with `app.role = 'internal'`, which bypasses the customer-scope filter but not other RLS.
- The model is instructed not to cross account boundaries; tool-layer enforcement makes this cryptographic, not just behavioral.

---

## Running the Proactive Scanner

For internal ops staff, the proactive scanner runs on demand:

```bash
python proactive/detector.py
```

Or integrate it into the Chainlit startup for internal sessions (already wired in `ui/app.py` comments).

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `MISTRAL_API_KEY` | ✅ | - | Mistral AI API key |
| `HF_TOKEN` | ✅ | - | HuggingFace Inference API token |
| `SUPABASE_URL` | ✅ | - | Supabase project URL |
| `SUPABASE_ANON_KEY` | ✅ | - | Supabase anon/public key |
| `SUPABASE_SERVICE_ROLE_KEY` | ✅ | - | Supabase service role key (ingestion only) |
| `HISTORY_WINDOW_HOURS` | ❌ | `1` | Hours of conversation to retain |
| `MAX_HISTORY_MESSAGES` | ❌ | `30` | Hard cap on message count |
| `RETRIEVAL_TOP_K` | ❌ | `5` | Number of document chunks per search |
