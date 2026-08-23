# ParcelPilot Support Agent - Demo Video Script
# CalQuity AI Engineer Assessment | ~5 minutes

---

## SEGMENT 1 - INTRO + PROBLEM STATEMENT (0:00 - 0:40)

**[Screen: README or a blank slide with "ParcelPilot Support Agent"]**

"Hi, I'm Sanskar. This is my submission for the CalQuity AI Engineer assessment.

The brief was to build an AI support system for ParcelPilot - a B2B logistics platform
whose 20-person ops team manually searches across policies, customer agreements,
product docs, past tickets, and live order data to resolve hundreds of support requests
a week.

The interesting part of the brief wasn't just 'build a chatbot'. It was:
- The source base is intentionally imperfect - some docs are outdated
- Customer agreements can override general policy
- Historical ticket answers may be wrong
- You need to handle these conflicts deliberately

And the two stretch problems: proactive issue detection, and building a system
the ops team will actually trust.

I built both user contexts - a customer portal and an internal ops interface -
deployed on Render at parcelpilot-agent.onrender.com."

---

## SEGMENT 2 - ARCHITECTURE OVERVIEW (0:40 - 2:00)

**[Screen: architecture diagram from README or a simple drawn diagram]**

"Let me walk through how it's built.

The entry point is a custom FastAPI + WebSocket server. I chose a custom UI over
Chainlit because the dual-auth flow - customer PIN vs staff password - needed
precise control over routing and session context. The WebSocket streams every
token as it arrives from the model.

Each user message goes through this path:
  1. Saved to Supabase conversation_messages
  2. Last 3 turns of history fetched back from DB
  3. A fresh LangGraph state is built with that history
  4. The graph runs: agent_node calls tools, tool_node executes them, loops back
  5. Response is streamed to the UI and saved to DB

The LangGraph graph is deliberately simple - two nodes in a loop.
No state machine. Complexity lives in the tools and the prompt, not the graph.

**Three tools:**

Tool 1 - search_documents: takes the user's query, embeds it via HuggingFace
BAAI/bge-small-en-v1.5, runs cosine similarity against Supabase pgvector.
Returns the top-5 chunks with source name and an authority level - 1 through 4.

Tool 2 - query_data: structured lookups against accounts, orders, tickets.
Intents: lookup_order, lookup_account, list_open_tickets, calculate_credit_eligibility,
check_sla_breach. For customer sessions, RLS policies enforce account scoping at
the database layer - the model can't talk its way past them.

Tool 3 - create_action: accepts an action type and payload plus an execute flag.
execute=false returns a draft for review. execute=true (after the user confirms)
commits to the actions_log table. Types: escalate_ticket, update_ticket_status,
create_followup_task.

For embeddings I used HuggingFace Inference API - no local GPU needed.
The PDFs are chunked at 600 tokens with 100-token overlap.
The xlsx maps to three structured tables: accounts, orders, tickets."

---

## SEGMENT 3 - LIVE DEMO (2:00 - 3:45)

**[Screen: browser at parcelpilot-agent.onrender.com]**

### Customer portal demo (2:00 - 2:45)

"I'll start with the customer portal. Logging in as Northstar Logistics - account
ACCT-001, PIN 1234.

Let me ask the example question from the brief: 'Can Northstar cancel ORD-1001
without a cancellation fee? Explain why.'

[submit query - watch streaming]

You can see the agent is chaining tools: it looks up the order first to confirm
it exists and check its status, then searches the Northstar Enterprise Agreement
for the cancellation clause, then cross-references the Cancellation SOP.

The key thing here: the Northstar agreement has a higher authority level than
the general SOP, so the agent applies the agreement terms and explicitly says why.
It's not guessing - it's citing the source hierarchy.

Now let me ask about a service credit - 'A pickup is three hours late because
of carrier fault. Should I get a service credit?'

[submit query]

This is a multi-step request: it looks up my open orders, calculates hours past
the pickup window, checks carrier_fault flag in the data, then reads the SOP
credit thresholds. Notice it also checks my agreement for any override first.
It won't promise a credit if the data doesn't support it."

### Internal ops demo - proactive scan + confirmation gate (2:45 - 3:45)

"Now switching to Internal Ops. Logging in as priya.

The first thing you see before I type anything is the proactive issue scan.
It fires on every internal login - no prompt needed. It's checking:
SLA breaches on open tickets, orders stuck in failed-pickup, and ticket clusters
detected by Jaccard similarity on subject keywords.

This is addressing CalQuity's first stretch problem: the ops team shouldn't have
to ask 'what needs my attention?' - the system should tell them.

Now let me demonstrate the confirmation gate. I'll ask to escalate a ticket:
'Escalate TKT-001 to P1.'

[submit query]

The agent calls create_action with execute=False first - that's enforced in the
system prompt and the tool design. It returns a draft: exactly what will happen,
which account, which severity. Then it asks 'Shall I go ahead?'

I reply 'yes'. On the next turn, the agent reads the DB-persisted conversation
history, sees the prior proposal and my confirmation, and calls create_action
with execute=True. The action is committed.

No in-memory state. No pending_action flag. The LLM reads history exactly like
a human reads an email thread."

---

## SEGMENT 4 - KEY DECISIONS + TRADE-OFFS (3:45 - 5:00)

**[Screen: code editor or slide]**

"Three decisions worth explaining:

**Decision 1 - DB-backed confirmation over a state machine.**

The obvious approach is a PENDING_CONFIRMATION state in the graph.
The problem: it ties confirmation to a running process. If the server restarts
between turns, the pending state is gone. It also means every 'yes' or 'no'
has to be routed specially.

The better approach: save every message to Supabase. On every turn, fetch
the last 3 exchanges. The agent reads the history and makes the decision -
the same way you'd infer 'they said yes' from an email chain.
This is more robust and requires no special routing. The trade-off is one
extra Supabase round-trip per turn - about 50ms.

**Decision 2 - RLS enforcement over model-instruction-only access control.**

The brief explicitly said: 'Access controls should be enforced in the data/tool
layer rather than relying only on model instructions.'

I take that seriously. Supabase RLS policies mean a SELECT on the orders table
under a customer session returns exactly zero rows for any other account,
regardless of what the prompt says. You can't prompt-inject your way
to ACCT-002's data when logged in as ACCT-001.

The Python layer also re-checks account_id before returning any row.
Two independent layers.

**Decision 3 - Source authority as a structured property, not vague prompt guidance.**

Each document chunk has an authority_level integer: 1 for customer agreements,
2 for current policy, 3 for SOPs, 4 for historical tickets.
The agent uses this number to resolve conflicts - lower number wins.

I also mark deprecated documents explicitly - the v2 policy is tagged is_deprecated=true
at ingestion. If it appears in retrieval results, the agent ignores it entirely.

What I left out: real carrier API integration, multi-language support, fine-tuning.
All deliberate - the goal was a reliable, explainable, deployable system within
the assessment timeframe.

The one metric I'd use to judge success: first-contact resolution rate.
What percentage of support queries get a complete answer from the agent
without a human follow-up? Anything above 60% on policy and SLA questions
would justify the ops time saved.

Code is at github.com/SanskarBajaj123/ParcelPilot.
Live demo at parcelpilot-agent.onrender.com.

Thanks."

---

## NOTES FOR RECORDING

- Total runtime target: 4:45 - 5:15
- Segment 1: speak at normal pace, no rushing - sets context for everything
- Segment 2: diagram on screen, talk through top-down; don't go line by line through code
- Segment 3: have the browser open at the hosted URL before recording; pre-type queries
  to save time; don't wait for full response before narrating what's happening
- Segment 4: this is the most important segment for CalQuity - they want product judgment,
  not just feature list; explain the WHY behind each decision
- Skip source chip / conflict chip - removed from UI; agent states conflicts verbally

## QUERIES TO PRE-TYPE (copy-paste during demo)

Customer portal (ACCT-001 / 1234):
- "Can Northstar cancel ORD-1001 without a cancellation fee? Explain why."
- "A pickup is three hours late because of carrier fault. Should I get a service credit?"

Internal ops (priya / ParcelPilot@2026):
- [observe proactive scan on login - no typing needed]
- "Escalate TKT-001 to P1."
- [wait for draft, then type]: "yes"
