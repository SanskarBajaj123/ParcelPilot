"""
Chainlit UI for ParcelPilot Support Agent.

On startup: role selector (customer / internal ops).
Customer mode: asks for account ID â†’ loads account name.
Internal mode: goes straight to chat + runs proactive issue scan.

Tool call steps are streamed live via Chainlit Step elements.
"""

import os
import time
import asyncio
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

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)


# â”€â”€ Startup: role selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@cl.on_chat_start
async def on_start():
    res = await cl.AskActionMessage(
        content="Welcome to **ParcelPilot Support**. How are you accessing this system?",
        actions=[
            cl.Action(name="customer",  label="Customer Portal",     value="customer"),
            cl.Action(name="internal",  label="Internal Support Ops", value="internal"),
        ],
        timeout=120,
    ).send()

    if not res:
        await cl.Message(content="Session timed out. Please refresh.").send()
        return

    role = res.get("value", "customer")
    if role == "customer":
        await _setup_customer()
    else:
        await _setup_internal()


async def _setup_customer():
    account_res = await cl.AskUserMessage(
        content="Please enter your **Account ID** (e.g. ACCT-001):",
        timeout=120,
    ).send()

    if not account_res:
        await cl.Message(content="No account ID provided. Please refresh and try again.").send()
        return

    raw_id = account_res["output"].strip().upper()

    rows = sb.table("accounts").select("account_id,account_name").eq("account_id", raw_id).execute().data
    if not rows:
        await cl.Message(
            content=f"Account **{raw_id}** not found. Please check your Account ID and refresh."
        ).send()
        return

    acct = rows[0]
    user_ctx: UserContext = {
        "account_id":   acct["account_id"],
        "account_name": acct["account_name"],
        "role":         "customer",
        "name":         acct["account_name"],
    }
    cl.user_session.set("user_context", user_ctx)
    cl.user_session.set("agent_state",  _initial_state(user_ctx))

    await cl.Message(
        content=(
            f"Hello, **{acct['account_name']}**!\n\n"
            "I'm the ParcelPilot Support Agent. I can help you with:\n"
            "- Order status and tracking\n"
            "- Service credits and cancellation queries\n"
            "- SLA and support ticket questions\n\n"
            "How can I help you today?"
        )
    ).send()


async def _setup_internal():
    name_res = await cl.AskUserMessage(
        content="Enter your name (for action logging):",
        timeout=60,
    ).send()

    staff_name = name_res["output"].strip() if name_res else "Support Staff"

    user_ctx: UserContext = {
        "account_id":   None,
        "account_name": None,
        "role":         "internal",
        "name":         staff_name,
    }
    cl.user_session.set("user_context", user_ctx)
    cl.user_session.set("agent_state",  _initial_state(user_ctx))

    await cl.Message(
        content=(
            f"Welcome, **{staff_name}**! Internal Ops mode active.\n\n"
            "You have access to all account data. I can help you:\n"
            "- Look up any order, account, or ticket\n"
            "- Check SLA breach status across accounts\n"
            "- Escalate tickets and create follow-up tasks\n"
            "- Search all policy and SOP documents\n\n"
            "Running proactive issue scanâ€¦"
        )
    ).send()

    # Run proactive scan in thread (it's synchronous Supabase calls)
    try:
        from proactive.detector import detect_issues
        result = await asyncio.get_event_loop().run_in_executor(None, detect_issues)
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


# â”€â”€ Message handler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@cl.on_message
async def on_message(message: cl.Message):
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

    # key: tool_call_id (from event metadata), value: cl.Step
    active_steps: dict[str, cl.Step] = {}
    final_text = ""
    new_state  = state

    try:
        async for event in graph.astream_events(new_state, version="v2"):
            kind = event["event"]
            tags = event.get("tags", [])
            name = event.get("name", "")
            data = event.get("data", {})

            # â”€â”€ LLM token streaming â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    token = chunk.content
                    if isinstance(token, str) and token:
                        final_text += token
                        await msg_placeholder.stream_token(token)

            # â”€â”€ Tool call started â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            elif kind == "on_tool_start":
                tool_id = event.get("run_id", name)
                step = cl.Step(name=name, type="tool")
                await step.__aenter__()
                active_steps[tool_id] = step
                inp = data.get("input", {})
                step.input = str(inp)[:300]

            # â”€â”€ Tool call finished â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            elif kind == "on_tool_end":
                tool_id = event.get("run_id", name)
                step = active_steps.pop(tool_id, None)
                if step:
                    out = data.get("output", "")
                    step.output = str(out)[:500] + ("â€¦" if len(str(out)) > 500 else "")
                    await step.__aexit__(None, None, None)

            # â”€â”€ Full graph output (grab final state) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # Sources panel
    sources = new_state.get("sources_used", [])
    if sources:
        lines = [
            f"- **{s['source_file']}** (authority {s['authority_level']}, p.{s.get('page_num','?')}): "
            f"{s.get('preview','')[:80]}â€¦"
            for s in sources[:5]
        ]
        await cl.Message(
            content="**Sources consulted:**\n" + "\n".join(lines),
            author="Sources",
        ).send()

    # Conflict warning
    if new_state.get("conflict_detected"):
        await cl.Message(
            content=(
                "âš ï¸ **Conflict detected between sources.**\n"
                "The higher-authority source (customer agreement) was applied. "
                "The agent has noted which source governs."
            ),
            author="Warning",
        ).send()

    cl.user_session.set("agent_state", new_state)

