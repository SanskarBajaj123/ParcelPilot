"""
LangGraph nodes: agent_node, tool_node, confirm_node.
"""

import os
import time
import json
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .state import AgentState
from .history import filter_history, stamp
from .prompts import build_system_prompt, build_retrieved_context_block
from .tools.search_documents import search_documents_fn
from .tools.query_data import query_data_fn
from .tools.create_action import create_action_fn

MISTRAL_API_KEY      = os.environ["MISTRAL_API_KEY"]
HISTORY_WINDOW_HOURS = float(os.getenv("HISTORY_WINDOW_HOURS", "1"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))

# ── Mistral LLM ───────────────────────────────────────────────────────────────
mistral_llm = ChatMistralAI(
    model="mistral-large-latest",
    api_key=MISTRAL_API_KEY,
    temperature=0.1,     # low temperature for factual support answers
    streaming=True,
)

# ── Tool schemas for Mistral function calling ────────────────────────────────
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search ParcelPilot policies, SOPs, customer agreements, and product documentation. Use before answering any policy or contractual question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or question to search for in the documents."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": "Look up structured data: orders, accounts, tickets. Calculate credit eligibility or SLA breach status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["lookup_order", "lookup_account", "list_open_tickets", "calculate_credit_eligibility", "check_sla_breach"],
                        "description": "The type of data query to perform."
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters. Examples: {order_id: 'ORD-1001'} for lookup_order; {ticket_id: 'TKT-501'} for check_sla_breach."
                    },
                },
                "required": ["intent", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_action",
            "description": "Create a support action (escalation, ticket update, follow-up task). Use execute=false to prepare a draft for user confirmation, then execute=true after explicit user consent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action_type": {
                        "type": "string",
                        "enum": ["escalate_ticket", "update_ticket_status", "create_followup_task"],
                    },
                    "payload": {
                        "type": "object",
                        "description": "Action-specific fields. escalate_ticket: {ticket_id, severity, reason}. update_ticket_status: {ticket_id, new_status, note}. create_followup_task: {ticket_id, description, due_hours}."
                    },
                    "execute": {
                        "type": "boolean",
                        "description": "false = draft only (no side effects). true = commit action. ONLY set true after explicit user confirmation.",
                    },
                },
                "required": ["action_type", "payload", "execute"],
            },
        },
    },
]


# ── agent_node ────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    Main LLM call. Assembles:
      1. System prompt (with user context)
      2. Retrieved context block (fresh search this turn, if a user message is latest)
      3. Time-filtered conversation history
    Binds tool schemas unless in PENDING_CONFIRMATION state.
    """
    user_ctx = state["user_context"]
    sys_prompt = build_system_prompt(user_ctx)

    # Time-window history filter
    history = filter_history(
        state["messages"],
        window_hours=HISTORY_WINDOW_HOURS,
        max_messages=MAX_HISTORY_MESSAGES,
    )

    # Retrieve relevant docs for the latest human message (if any)
    retrieved_ctx_block = ""
    latest_human = next(
        (m for m in reversed(history) if isinstance(m, HumanMessage)), None
    )
    if latest_human:
        account_scope = user_ctx.get("account_id") if user_ctx["role"] == "customer" else None
        search_result = search_documents_fn(
            query=latest_human.content if isinstance(latest_human.content, str) else str(latest_human.content),
            account_scope=account_scope,
        )
        retrieved_ctx_block = build_retrieved_context_block(
            search_result["chunks"],
            search_result["conflict_detected"],
        )

    # Build message list for LLM
    messages_for_llm = [SystemMessage(content=sys_prompt)]
    if retrieved_ctx_block:
        messages_for_llm.append(SystemMessage(content=retrieved_ctx_block))
    messages_for_llm.extend(history)

    # Disable tool calls while waiting for confirmation
    if state["confirmation_state"] == "PENDING_CONFIRMATION":
        llm = mistral_llm   # no tools bound
    else:
        llm = mistral_llm.bind_tools(TOOL_SCHEMAS)

    response = llm.invoke(messages_for_llm)
    stamped  = stamp(response)

    return {
        "messages":             [stamped],
        "tool_calls_this_turn": [tc["name"] for tc in (response.tool_calls or [])],
        "sources_used":         [],
        "conflict_detected":    False,
    }


# ── tool_node ─────────────────────────────────────────────────────────────────

def tool_node(state: AgentState) -> dict:
    """
    Dispatches tool calls from the latest AIMessage.
    Injects user_context into every tool call — model cannot override access control.
    """
    user_ctx    = state["user_context"]
    role        = user_ctx["role"]
    account_id  = user_ctx.get("account_id") or ""
    actor_name  = user_ctx.get("name", "unknown")

    last_ai     = state["messages"][-1]
    tool_msgs   = []
    all_sources = []
    conflict    = False

    for tc in (last_ai.tool_calls or []):
        name   = tc["name"]
        args   = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
        tc_id  = tc["id"]

        # ── search_documents ─────────────────────────────────────────────────
        if name == "search_documents":
            account_scope = account_id if role == "customer" else None
            result = search_documents_fn(
                query=args.get("query", ""),
                account_scope=account_scope,
            )
            all_sources.extend(result.get("sources", []))
            if result.get("conflict_detected"):
                conflict = True

            # Format chunks as text for the ToolMessage
            chunks = result.get("chunks", [])
            content = build_retrieved_context_block(chunks, result["conflict_detected"])
            tool_msgs.append(stamp(ToolMessage(content=content, tool_call_id=tc_id)))

        # ── query_data ───────────────────────────────────────────────────────
        elif name == "query_data":
            result = query_data_fn(
                intent=args.get("intent", ""),
                params=args.get("params", {}),
                role=role,
                account_id=account_id,
            )
            tool_msgs.append(stamp(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tc_id)))

        # ── create_action ────────────────────────────────────────────────────
        elif name == "create_action":
            execute = args.get("execute", False)

            # Safety: if somehow execute=True without PENDING_CONFIRMATION, block it
            if execute and state["confirmation_state"] != "PENDING_CONFIRMATION":
                tool_msgs.append(stamp(ToolMessage(
                    content='{"error": "Cannot execute action without pending confirmation. Call create_action with execute=false first to prepare a draft."}',
                    tool_call_id=tc_id,
                )))
                continue

            result = create_action_fn(
                action_type=args.get("action_type", ""),
                payload=args.get("payload", {}),
                execute=execute,
                actor_name=actor_name,
                actor_role=role,
            )

            if result.get("draft"):
                # Set confirmation gate
                pending = {
                    "action_type": result["action_type"],
                    "payload":     result["payload"],
                    "summary":     result["summary"],
                }
                return {
                    "messages":             [stamp(ToolMessage(content=result["summary"], tool_call_id=tc_id))],
                    "confirmation_state":   "PENDING_CONFIRMATION",
                    "pending_action":       pending,
                    "sources_used":         all_sources,
                    "conflict_detected":    conflict,
                }

            tool_msgs.append(stamp(ToolMessage(content=json.dumps(result), tool_call_id=tc_id)))

        else:
            tool_msgs.append(stamp(ToolMessage(
                content=f'{{"error": "Unknown tool: {name}"}}',
                tool_call_id=tc_id,
            )))

    return {
        "messages":          tool_msgs,
        "sources_used":      all_sources,
        "conflict_detected": conflict,
    }


# ── confirm_node ──────────────────────────────────────────────────────────────

def confirm_node(state: AgentState) -> dict:
    """
    Handles the user's yes/no response to a pending action.
    Called when confirmation_state == PENDING_CONFIRMATION and a new HumanMessage arrives.
    """
    user_ctx   = state["user_context"]
    pending    = state["pending_action"]
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )

    if not last_human or not pending:
        return {}

    user_reply = last_human.content.strip().lower() if isinstance(last_human.content, str) else ""
    confirmed  = any(w in user_reply for w in ["yes", "confirm", "go ahead", "proceed", "ok", "sure", "yeah", "yep"])
    cancelled  = any(w in user_reply for w in ["no", "cancel", "stop", "don't", "dont", "nope", "abort"])

    if confirmed:
        result = create_action_fn(
            action_type=pending["action_type"],
            payload=pending["payload"],
            execute=True,
            actor_name=user_ctx.get("name", "unknown"),
            actor_role=user_ctx["role"],
        )
        reply = AIMessage(content=f"Done. Action completed.\n\n{result.get('summary', '')}\nAction ID: {result.get('action_id', 'N/A')}")
        return {
            "messages":           [stamp(reply)],
            "confirmation_state": "IDLE",
            "pending_action":     None,
        }

    elif cancelled:
        reply = AIMessage(content="Understood — action cancelled. Let me know if you'd like to do something else.")
        return {
            "messages":           [stamp(reply)],
            "confirmation_state": "IDLE",
            "pending_action":     None,
        }

    else:
        # Ambiguous — re-ask
        reply = AIMessage(
            content=(
                f"I need a clear yes or no to proceed.\n\n"
                f"**Pending action:**\n{pending['summary']}\n\n"
                "Reply **yes** to confirm or **no** to cancel."
            )
        )
        return {"messages": [stamp(reply)]}
