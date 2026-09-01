"""
LangGraph nodes: agent_node, tool_node.
"""

import os
import json
import time
import logging
from langsmith import traceable
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from .state import AgentState
from .history import filter_history, stamp
from .prompts import build_system_prompt, build_retrieved_context_block
from .tools.search_documents import search_documents_fn
from .tools.query_data import query_data_fn
from .tools.create_action import create_action_fn

logger = logging.getLogger(__name__)

MISTRAL_API_KEY   = os.environ["MISTRAL_API_KEY"]
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))

# ── Mistral LLM ───────────────────────────────────────────────────────────────
mistral_llm = ChatMistralAI(
    model="open-mixtral-8x22b",
    api_key=MISTRAL_API_KEY,
    temperature=0.1,
    streaming=True,
)

# ── Rate-limit aware invoke ───────────────────────────────────────────────────
def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg

def _is_retryable(exc: Exception) -> bool:
    return _is_rate_limit(exc) or "500" in str(exc) or "503" in str(exc)

@traceable(name="LLM: Mistral Large 2", run_type="llm")
@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=2, max=64),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _invoke_llm(llm, messages: list) -> AIMessage:
    try:
        return llm.invoke(messages)
    except Exception as exc:
        if _is_retryable(exc):
            logger.warning("Mistral API error (%s) - will retry with backoff", exc)
            raise
        raise

@traceable(name="Preemptive: Document Search", run_type="retriever")
def _preemptive_search(query: str, account_scope: str | None) -> dict:
    return search_documents_fn(query=query, account_scope=account_scope)


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

@traceable(name="Agent: Reasoning", run_type="chain")
def agent_node(state: AgentState) -> dict:
    """
    Main LLM call. Assembles:
      1. System prompt (with user context)
      2. Retrieved context block (fresh search this turn, if a user message is latest)
      3. Conversation history (from DB, injected via state["messages"])
    """
    user_ctx   = state["user_context"]
    sys_prompt = build_system_prompt(user_ctx)

    # Keep last MAX_HISTORY_TURNS complete turns from the injected history
    history = filter_history(
        state["messages"],
        max_turns=MAX_HISTORY_TURNS,
    )

    # Retrieve relevant docs for the latest human message (if any).
    # Skip on the second agent call (after tool_node ran) - tool results are already in history.
    retrieved_ctx_block = ""
    preemptive_sources  = []
    preemptive_conflict = False
    last_msg      = state["messages"][-1] if state["messages"] else None
    is_after_tools = isinstance(last_msg, ToolMessage)

    if not is_after_tools:
        latest_human = next(
            (m for m in reversed(history) if isinstance(m, HumanMessage)), None
        )
        if latest_human:
            role          = user_ctx["role"]
            account_scope = user_ctx.get("account_id") if role == "customer" else None
            account_name  = user_ctx.get("account_name", "")
            raw_query     = latest_human.content if isinstance(latest_human.content, str) else str(latest_human.content)

            enhanced_query = f"{raw_query} {account_name} agreement" if (role == "customer" and account_name) else raw_query
            search_result  = _preemptive_search(query=enhanced_query, account_scope=account_scope)

            if role == "customer" and account_name and not any(c["authority_level"] == 1 for c in search_result["chunks"]):
                targeted = _preemptive_search(
                    query=f"{account_name} service agreement SLA priority response time credit cancellation",
                    account_scope=account_scope,
                )
                existing = {c.get("id") for c in search_result["chunks"]}
                for chunk in targeted["chunks"]:
                    if chunk["authority_level"] == 1 and chunk.get("id") not in existing:
                        search_result["chunks"].append(chunk)
                if targeted["conflict_detected"]:
                    search_result["conflict_detected"] = True
                search_result["chunks"].sort(key=lambda c: (c["authority_level"], -c.get("similarity", 0)))

            retrieved_ctx_block = build_retrieved_context_block(
                search_result["chunks"],
                search_result["conflict_detected"],
            )
            preemptive_sources  = search_result.get("sources", [])
            preemptive_conflict = search_result["conflict_detected"]

    messages_for_llm = [SystemMessage(content=sys_prompt)]
    if retrieved_ctx_block:
        messages_for_llm.append(SystemMessage(content=retrieved_ctx_block))
    messages_for_llm.extend(history)

    llm      = mistral_llm.bind_tools(TOOL_SCHEMAS)
    response = _invoke_llm(llm, messages_for_llm)
    stamped  = stamp(response)

    out: dict = {
        "messages":             [stamped],
        "tool_calls_this_turn": [tc["name"] for tc in (response.tool_calls or [])],
    }
    if not is_after_tools:
        out["sources_used"]      = preemptive_sources
        out["conflict_detected"] = preemptive_conflict
    return out


# ── tool_node ─────────────────────────────────────────────────────────────────

@traceable(name="Tools: Dispatch", run_type="chain")
def tool_node(state: AgentState) -> dict:
    """
    Dispatches tool calls from the latest AIMessage.
    Injects user_context into every tool call -- model cannot override access control.
    """
    user_ctx   = state["user_context"]
    role       = user_ctx["role"]
    account_id = user_ctx.get("account_id") or ""
    actor_name = user_ctx.get("name", "unknown")

    last_ai   = state["messages"][-1]
    tool_msgs = []
    all_sources: list = []
    conflict = False

    for tc in (last_ai.tool_calls or []):
        name  = tc["name"]
        args  = tc["args"] if isinstance(tc["args"], dict) else json.loads(tc["args"])
        tc_id = tc["id"]

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

            chunks = result.get("chunks", [])
            if not chunks:
                content = (
                    "TOOL_RESULT: NO DOCUMENTS FOUND\n"
                    "The document search returned no relevant results for this query. "
                    "Do NOT guess or use prior conversation context to answer. "
                    "Tell the user that no relevant policy or document information was found."
                )
            else:
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
            is_empty = (
                "error" in result
                or result.get("result") == "NO_DATA"
                or (isinstance(result.get("tickets"), list) and len(result["tickets"]) == 0)
                or (isinstance(result.get("orders"), list) and len(result["orders"]) == 0)
            )
            if is_empty:
                content = (
                    "TOOL_RESULT: NO DATA FOUND\n"
                    "The database returned no records for this query. "
                    "Do NOT infer, guess, or use any previous conversation context to fill in missing data. "
                    "Tell the user exactly what was searched and that no data is available.\n\n"
                    f"Raw result: {json.dumps(result, default=str)}"
                )
            else:
                content = json.dumps(result, default=str)
            tool_msgs.append(stamp(ToolMessage(content=content, tool_call_id=tc_id)))

        # ── create_action ────────────────────────────────────────────────────
        elif name == "create_action":
            execute = args.get("execute", False)

            result = create_action_fn(
                action_type=args.get("action_type", ""),
                payload=args.get("payload", {}),
                execute=execute,
                actor_name=actor_name,
                actor_role=role,
            )
            tool_msgs.append(stamp(ToolMessage(
                content=result.get("summary", json.dumps(result)),
                tool_call_id=tc_id,
            )))

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
