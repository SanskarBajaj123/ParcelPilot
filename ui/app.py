"""
Chainlit UI for ParcelPilot Support Agent.

On startup: role selector (customer / internal ops).
Customer mode: asks for account ID â†' loads account name.
Internal mode: goes straight to chat + runs proactive issue scan.

Tool call steps are streamed live via Chainlit Step elements.
"""

import os
import time
import hashlib
import asyncio
from datetime import datetime, timezone
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from supabase import create_client

from pathlib import Path as _Path
load_dotenv(_Path(__file__).parent.parent / ".env" if (_Path(__file__).parent.parent / ".env").exists() else _Path(__file__).parent.parent.parent / ".env", override=True)

# Import after load_dotenv so env vars are available at module-level in nodes.py
from agent.graph import graph
from agent.state import AgentState, UserContext
from agent.history import stamp

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_ANON_KEY"]
SUPABASE_SVC  = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb     = create_client(SUPABASE_URL, SUPABASE_KEY)
sb_svc = create_client(SUPABASE_URL, SUPABASE_SVC)

MAX_AUTH_ATTEMPTS = 3

# Demo customer PINs -- in production: hashed in accounts table, verified server-side
_CUSTOMER_PINS: dict[str, str] = {
    "ACCT-001": "1234",
    "ACCT-002": "5678",
    "ACCT-003": "4321",
    "ACCT-004": "8765",
}

def _h(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# Staff username -> SHA-256(password).  In production: db lookup with bcrypt.
_STAFF_CREDS: dict[str, str] = {
    "priya": _h("ParcelPilot@2026"),
    "rohit": _h("Support@2026!"),
    "maya":  _h("Ops#Secure99"),
    "admin": _h("Admin$Master1"),
}
_STAFF_NAMES: dict[str, str] = {
    "priya": "Priya Nair",
    "rohit": "Rohit Sharma",
    "maya":  "Maya Pillai",
    "admin": "Administrator",
}

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _append_session_log(event_type: str, detail: str, latency_ms: int | None = None) -> None:
    """Append an event to the per-session activity log stored in cl.user_session."""
    entry: dict = {"time": _ts(), "type": event_type, "detail": detail}
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    try:
        logs: list = cl.user_session.get("session_log") or []
        logs.append(entry)
        cl.user_session.set("session_log", logs)
    except Exception:
        pass  # user_session not yet initialised during on_start


def _log_auth(event: str, detail: str) -> None:
    """Write auth events to server log and session log."""
    import logging
    logging.getLogger("parcelpilot.auth").info("[AUTH] %s | %s", event, detail)
    _append_session_log("AUTH", f"{event} | {detail}")


# â"€â"€ Startup: role selection â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

@cl.on_chat_start
async def on_start():
    res = await cl.AskActionMessage(
        content="Welcome to **ParcelPilot Support**. How are you accessing this system?",
        actions=[
            cl.Action(name="customer",  label="Customer Portal",     payload={"role": "customer"}),
            cl.Action(name="internal",  label="Internal Support Ops", payload={"role": "internal"}),
        ],
        timeout=120,
    ).send()

    if not res:
        await cl.Message(content="Session timed out. Please refresh.").send()
        return

    role = res.get("payload", {}).get("role", "customer")
    if role == "customer":
        await _setup_customer()
    else:
        await _setup_internal()


async def _setup_customer():
    # ── Step 1: Account ID ──────────────────────────────────────────────────────
    account_res = await cl.AskUserMessage(
        content="Please enter your **Account ID** (e.g. ACCT-001):",
        timeout=120,
    ).send()
    if not account_res:
        await cl.Message(content="No account ID provided. Please refresh and try again.").send()
        return

    raw_id = account_res["output"].strip().upper()
    rows = sb_svc.table("accounts").select("account_id,account_name").eq("account_id", raw_id).execute().data
    if not rows:
        _log_auth("CUSTOMER_FAIL", f"unknown account {raw_id}")
        await cl.Message(
            content=f"Account **{raw_id}** not found. Please check your Account ID and refresh."
        ).send()
        return

    acct = rows[0]

    # ── Step 2: PIN verification (up to MAX_AUTH_ATTEMPTS tries) ───────────────
    expected_pin = _CUSTOMER_PINS.get(acct["account_id"])
    if expected_pin is None:
        # Account exists but no PIN registered -- block access
        _log_auth("CUSTOMER_FAIL", f"no PIN configured for {acct['account_id']}")
        await cl.Message(
            content="Authentication is not configured for this account. "
                    "Please contact ParcelPilot support."
        ).send()
        return

    for attempt in range(1, MAX_AUTH_ATTEMPTS + 1):
        pin_res = await cl.AskUserMessage(
            content=(
                f"Enter your **4-digit security PIN** for account {acct['account_id']}"
                + (f" (attempt {attempt}/{MAX_AUTH_ATTEMPTS}):" if attempt > 1 else ":")
            ),
            timeout=120,
        ).send()

        if not pin_res:
            await cl.Message(content="PIN entry timed out. Please refresh.").send()
            return

        entered = pin_res["output"].strip()
        if entered == expected_pin:
            _log_auth("CUSTOMER_OK", acct["account_id"])
            break
        else:
            _log_auth("CUSTOMER_PIN_FAIL", f"{acct['account_id']} attempt {attempt}")
            if attempt < MAX_AUTH_ATTEMPTS:
                await cl.Message(
                    content=f"Incorrect PIN. {MAX_AUTH_ATTEMPTS - attempt} attempt(s) remaining."
                ).send()
            else:
                await cl.Message(
                    content="Too many incorrect PIN attempts. Access denied -- please refresh and try again."
                ).send()
                return

    user_ctx: UserContext = {
        "account_id":   acct["account_id"],
        "account_name": acct["account_name"],
        "role":         "customer",
        "name":         acct["account_name"],
    }
    cl.user_session.set("user_context", user_ctx)
    cl.user_session.set("agent_state",  _initial_state(user_ctx))
    cl.user_session.set("auth_verified", True)

    _append_session_log("AUTH", f"CUSTOMER_OK | {acct['account_id']} logged in")
    await cl.Message(
        content=(
            f"Hello, **{acct['account_name']}**!\n\n"
            "I'm the ParcelPilot Support Agent. I can help you with:\n"
            "- Order status and tracking\n"
            "- Service credits and cancellation queries\n"
            "- SLA and support ticket questions\n\n"
            "How can I help you today?"
        ),
        actions=[
            cl.Action(name="mock_accounts", label="Mock Accounts", payload={}, description="View test credentials"),
            cl.Action(name="view_logs",     label="Session Logs",  payload={}, description="View live session activity"),
        ],
    ).send()


async def _setup_internal():
    # ── Step 1: Username ────────────────────────────────────────────────────────
    user_res = await cl.AskUserMessage(
        content="**Internal Staff Login**\n\nEnter your staff username:",
        timeout=60,
    ).send()
    if not user_res:
        await cl.Message(content="Login timed out. Please refresh.").send()
        return

    username = user_res["output"].strip().lower()
    if username not in _STAFF_CREDS:
        _log_auth("STAFF_FAIL", f"unknown username '{username}'")
        await cl.Message(content="Username not recognised. Please refresh and try again.").send()
        return

    # ── Step 2: Password (up to MAX_AUTH_ATTEMPTS tries) ───────────────────────
    for attempt in range(1, MAX_AUTH_ATTEMPTS + 1):
        pw_res = await cl.AskUserMessage(
            content=(
                f"Enter your **password**"
                + (f" (attempt {attempt}/{MAX_AUTH_ATTEMPTS}):" if attempt > 1 else ":")
            ),
            timeout=60,
        ).send()
        if not pw_res:
            await cl.Message(content="Login timed out. Please refresh.").send()
            return

        if _h(pw_res["output"]) == _STAFF_CREDS[username]:
            _log_auth("STAFF_OK", username)
            break
        else:
            _log_auth("STAFF_PW_FAIL", f"username '{username}' attempt {attempt}")
            if attempt < MAX_AUTH_ATTEMPTS:
                await cl.Message(
                    content=f"Incorrect password. {MAX_AUTH_ATTEMPTS - attempt} attempt(s) remaining."
                ).send()
            else:
                await cl.Message(
                    content="Too many failed attempts. Access denied -- please refresh and try again."
                ).send()
                return

    staff_name = _STAFF_NAMES.get(username, username.title())

    user_ctx: UserContext = {
        "account_id":   None,
        "account_name": None,
        "role":         "internal",
        "name":         staff_name,
    }
    cl.user_session.set("user_context", user_ctx)
    cl.user_session.set("agent_state",  _initial_state(user_ctx))
    cl.user_session.set("auth_verified", True)

    _append_session_log("AUTH", f"STAFF_OK | {username} ({staff_name}) logged in")
    await cl.Message(
        content=(
            f"Welcome, **{staff_name}**! Internal Ops mode active.\n\n"
            "You have access to all account data. I can help you:\n"
            "- Look up any order, account, or ticket\n"
            "- Check SLA breach status across accounts\n"
            "- Escalate tickets and create follow-up tasks\n"
            "- Search all policy and SOP documents\n\n"
            "Running proactive issue scan..."
        ),
        actions=[
            cl.Action(name="mock_accounts", label="Mock Accounts", payload={}, description="View test credentials"),
            cl.Action(name="view_logs",     label="Session Logs",  payload={}, description="View live session activity"),
        ],
    ).send()

    # Run proactive scan in thread (it's synchronous Supabase calls)
    try:
        from proactive.detector import detect_issues
        t0 = time.time()
        result = await asyncio.get_event_loop().run_in_executor(None, detect_issues)
        lat = int((time.time() - t0) * 1000)
        breaches = len(result.get("sla_breaches", []))
        clusters = len(result.get("ticket_clusters", []))
        _append_session_log(
            "PROACTIVE_SCAN",
            f"{breaches} SLA breach(es), {clusters} cluster(s)",
            latency_ms=lat,
        )
        await cl.Message(content=result["summary"], author="Proactive Scan").send()
    except Exception as e:
        await cl.Message(
            content=f"Proactive scan unavailable: {e}\n\nYou can still query manually.",
            author="Proactive Scan",
        ).send()


def _initial_state(user_ctx: UserContext) -> AgentState:
    return {
        "messages":             [],
        "message_timestamps":   [],
        "user_context":         user_ctx,
        "confirmation_state":   "IDLE",
        "pending_action":       None,
        "sources_used":         [],
        "conflict_detected":    False,
        "tool_calls_this_turn": [],
    }


# â"€â"€ Message handler â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

@cl.on_message
async def on_message(message: cl.Message):
    # Guard: reject messages if auth was never completed
    if not cl.user_session.get("auth_verified"):
        await cl.Message(content="Session not authenticated. Please refresh.").send()
        return

    # -- Intercept header-button slash commands (don't log or send to LLM) ----
    cmd = message.content.strip()
    if cmd == "/__mock_accounts":
        await _show_mock_accounts()
        return
    if cmd == "/__logs":
        await _show_logs()
        return

    # All other messages: log and route to the LLM

    user_ctx: UserContext = cl.user_session.get("user_context")
    state: AgentState    = cl.user_session.get("agent_state")

    if not user_ctx or not state:
        await cl.Message(content="Session lost. Please refresh.").send()
        return

    # Append user message with timestamp
    human_msg = stamp(HumanMessage(content=message.content))
    state["messages"].append(human_msg)
    state["message_timestamps"].append(time.time())

    msg_placeholder = cl.Message(content="")
    await msg_placeholder.send()

    _append_session_log("USER_MSG", message.content[:120])

    # key: tool_call_id (from event metadata), value: cl.Step
    active_steps: dict[str, cl.Step] = {}
    tool_start_times: dict[str, float] = {}
    final_text     = ""
    new_state      = state
    captured_sources  = []   # accumulated across tool/agent node events
    captured_conflict = False
    llm_start_time: float | None = None

    try:
        async for event in graph.astream_events(new_state, version="v2"):
            kind = event["event"]
            tags = event.get("tags", [])
            name = event.get("name", "")
            data = event.get("data", {})

            # -- LLM token streaming -----------------------------------------------
            if kind == "on_chat_model_stream":
                if llm_start_time is None:
                    llm_start_time = time.time()
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    token = chunk.content
                    if isinstance(token, str) and token:
                        final_text += token
                        await msg_placeholder.stream_token(token)

            elif kind == "on_chat_model_start":
                llm_start_time = time.time()

            elif kind == "on_chat_model_end":
                if llm_start_time is not None:
                    lat = int((time.time() - llm_start_time) * 1000)
                    resp = data.get("output")
                    n_tool_calls = len(getattr(resp, "tool_calls", None) or [])
                    detail = f"LLM response: {len(final_text)} chars"
                    if n_tool_calls:
                        detail += f", {n_tool_calls} tool call(s) requested"
                    _append_session_log("LLM_CALL", detail, latency_ms=lat)
                    llm_start_time = None

            # -- Tool call started -------------------------------------------------
            elif kind == "on_tool_start":
                tool_id = event.get("run_id", name)
                step = cl.Step(name=name, type="tool")
                await step.__aenter__()
                active_steps[tool_id] = step
                tool_start_times[tool_id] = time.time()
                inp = data.get("input", {})
                step.input = str(inp)[:300]
                _append_session_log("TOOL_START", f"{name} | input: {str(inp)[:80]}")

            # -- Tool call finished ------------------------------------------------
            elif kind == "on_tool_end":
                tool_id = event.get("run_id", name)
                step = active_steps.pop(tool_id, None)
                lat = int((time.time() - tool_start_times.pop(tool_id, time.time())) * 1000)
                if step:
                    out = data.get("output", "")
                    step.output = str(out)[:500] + ("..." if len(str(out)) > 500 else "")
                    await step.__aexit__(None, None, None)
                _append_session_log("TOOL_END", f"{name}: done", latency_ms=lat)

            # -- Capture sources from node outputs ---------------------------------
            elif kind == "on_chain_end" and name in ("tool_node", "agent_node"):
                node_out = data.get("output", {})
                if node_out.get("sources_used"):
                    captured_sources  = node_out["sources_used"]
                    captured_conflict = node_out.get("conflict_detected", False)
                    src_names = [s["source_file"] for s in captured_sources[:3]]
                    _append_session_log("RETRIEVAL", f"{len(captured_sources)} sources: {', '.join(src_names)}")

            # â"€â"€ Full graph output (grab final state) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
            elif kind == "on_chain_end" and name == "LangGraph":
                new_state = data.get("output", new_state)

    except Exception as exc:
        await msg_placeholder.update()
        await cl.Message(content=f"**Error:** {exc}\n\nPlease try again.").send()
        # Close any open tool steps on error
        for step in active_steps.values():
            await step.__aexit__(None, None, None)
        return

    # Close any steps that didn't receive an end event
    for step in active_steps.values():
        await step.__aexit__(None, None, None)

    # If nothing was streamed (e.g. confirm_node wrote directly), pull from state
    if not final_text:
        last = new_state["messages"][-1] if new_state.get("messages") else None
        if isinstance(last, AIMessage) and isinstance(last.content, str):
            final_text = last.content
    msg_placeholder.content = final_text
    await msg_placeholder.update()

    # Sources panel: prefer state value, fall back to node-level captures
    sources = new_state.get("sources_used") or captured_sources
    conflict_flag = new_state.get("conflict_detected") or captured_conflict
    if sources:
        lines = [
            f"- **{s['source_file']}** (authority {s['authority_level']}, p.{s.get('page_num','?')}): "
            f"{s.get('preview','')[:80]}..."
            for s in sources[:5]
        ]
        await cl.Message(
            content="**Sources consulted:**\n" + "\n".join(lines),
            author="Sources",
        ).send()

    # Conflict warning
    if conflict_flag:
        await cl.Message(
            content=(
                "[CONFLICT] **Conflict detected between sources.**\n"
                "The higher-authority source (customer agreement) was applied. "
                "The agent has noted which source governs."
            ),
            author="Warning",
        ).send()

    _append_session_log("AGENT_REPLY", (final_text or "")[:120])
    cl.user_session.set("agent_state", new_state)


# -- Shared helpers (called by header buttons and action callbacks) ---------------

async def _show_mock_accounts():
    content = (
        "### Mock Account Details\n\n"
        "**Customer Accounts**\n\n"
        "| Account ID | PIN |\n"
        "|-----------|-----|\n"
        "| ACCT-001 | 1234 |\n"
        "| ACCT-002 | 5678 |\n"
        "| ACCT-003 | 4321 |\n"
        "| ACCT-004 | 8765 |\n\n"
        "**Staff Accounts (Internal Ops)**\n\n"
        "| Username | Password |\n"
        "|---------|----------|\n"
        "| priya | ParcelPilot@2026 |\n"
        "| rohit | Support@2026! |\n"
        "| maya | Ops#Secure99 |\n"
        "| admin | Admin$Master1 |"
    )
    await cl.Message(content=content, author="Mock Accounts").send()


async def _show_logs():
    logs: list = cl.user_session.get("session_log") or []
    if not logs:
        await cl.Message(content="No events recorded yet.", author="Session Logs").send()
        return

    # Group events into conversation turns (each USER_MSG starts a new turn)
    turns: list[list[dict]] = []
    current: list[dict] = []
    for entry in logs:
        if entry["type"] == "USER_MSG" and current:
            turns.append(current)
            current = []
        current.append(entry)
    if current:
        turns.append(current)

    lines = ["## Session Trace\n"]

    # Pre-session events (AUTH, PROACTIVE_SCAN) before the first user message
    pre = [e for e in logs if e["type"] in ("AUTH", "PROACTIVE_SCAN")]
    if pre:
        lines.append("### Session Start")
        for e in pre:
            lat = f" `{e['latency_ms']}ms`" if "latency_ms" in e else ""
            lines.append(f"- **{e['type']}**: {e['detail'][:100]}{lat}")
        lines.append("")

    for i, turn in enumerate(turns, 1):
        user_msg   = next((e for e in turn if e["type"] == "USER_MSG"), None)
        tool_calls = [e for e in turn if e["type"] in ("TOOL_START", "TOOL_END")]
        llm_calls  = [e for e in turn if e["type"] == "LLM_CALL"]
        retrieval  = [e for e in turn if e["type"] == "RETRIEVAL"]
        reply      = next((e for e in turn if e["type"] == "AGENT_REPLY"), None)

        q = user_msg["detail"][:80] if user_msg else "?"
        total_lat  = sum(e.get("latency_ms", 0) for e in turn)

        lines.append(f"### Turn {i} | `{user_msg['time'] if user_msg else '?'}` | {total_lat}ms total")
        lines.append(f"> **User:** {q}")
        lines.append("")

        if retrieval:
            for r in retrieval:
                lines.append(f"- **RETRIEVAL**: {r['detail'][:100]}")

        # Tool calls: match START/END pairs
        starts = [e for e in tool_calls if e["type"] == "TOOL_START"]
        ends   = {e["detail"].split(":")[0]: e for e in tool_calls if e["type"] == "TOOL_END"}
        for s in starts:
            tool_name = s["detail"].split(" |")[0]
            inp = s["detail"].split("input: ")[-1][:60] if "input:" in s["detail"] else ""
            end_key = tool_name
            end_e   = ends.get(end_key)
            lat_str = f" `{end_e['latency_ms']}ms`" if end_e and "latency_ms" in end_e else ""
            lines.append(f"- **TOOL** `{tool_name}`{lat_str} | input: _{inp}_")

        for lc in llm_calls:
            lat_str = f" `{lc['latency_ms']}ms`" if "latency_ms" in lc else ""
            lines.append(f"- **LLM**{lat_str}: {lc['detail'][:80]}")

        if reply:
            lines.append(f"- **Response**: _{reply['detail'][:80]}_")

        lines.append("")

    await cl.Message(content="\n".join(lines), author="Session Logs").send()


# -- Action callbacks (for message-level buttons, kept for backward compat) ------

@cl.action_callback("mock_accounts")
async def on_mock_accounts(action: cl.Action):
    if not cl.user_session.get("auth_verified"):
        await cl.Message(content="Please complete authentication first.").send()
        return
    await _show_mock_accounts()


@cl.action_callback("view_logs")
async def on_view_logs(action: cl.Action):
    if not cl.user_session.get("auth_verified"):
        await cl.Message(content="Please complete authentication first.").send()
        return
    await _show_logs()

