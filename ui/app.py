"""
Chainlit UI for ParcelPilot Support Agent.

On startup: role selector (customer / internal ops).
Customer mode: asks for account ID → loads account name.
Internal mode: goes straight to chat.

Tool call steps are streamed live via Chainlit Step elements.
"""

import os
import time
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Import the compiled LangGraph and state types
from agent.graph import graph
from agent.state import AgentState, UserContext
from agent.history import stamp

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)


# ── Startup: role selection ───────────────────────────────────────────────────

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

    # Verify account exists
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
            f"I'm the ParcelPilot Support Agent. I can help you with:\n"
            f"- Order status and tracking\n"
            f"- Service credits and cancellation queries\n"
            f"- SLA and support ticket questions\n\n"
            f"How can I help you today?"
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
            f"You have access to all account data. I can help you:\n"
            f"- Look up any order, account, or ticket\n"
            f"- Check SLA breach status across accounts\n"
            f"- Escalate tickets and create follow-up tasks\n"
            f"- Search all policy and SOP documents\n\n"
            f"What would you like to do?"
        )
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


# ── Message handler ───────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message):
    user_ctx: UserContext = cl.user_session.get("user_context")
    state: AgentState    = cl.user_session.get("agent_state")

    if not user_ctx or not state:
        await cl.Message(content="Session lost. Please refresh.").send()
        return

    # Add user message to state
    human_msg = stamp(HumanMessage(content=message.content))
    state["messages"].append(human_msg)
    state["message_timestamps"].append(time.time())

    # Run graph
    msg_placeholder = cl.Message(content="")
    await msg_placeholder.send()

    tool_steps: dict[str, cl.Step] = {}
    final_text  = ""
    new_state   = state

    try:
        # Stream events from LangGraph
        async for event in graph.astream_events(new_state, version="v2"):
            kind = event["event"]
            name = event.get("name", "")

            # ── Agent streaming ───────────────────────────────────────────────
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and isinstance(chunk.content, str):
                    final_text += chunk.content
                    await msg_placeholder.stream_token(chunk.content)

            # ── Tool call started ─────────────────────────────────────────────
            elif kind == "on_tool_start":
                step = cl.Step(name=f"Tool: {name}", type="tool")
                await step.__aenter__()
                tool_steps[name] = step
                input_data = event.get("data", {}).get("input", {})
                step.input = str(input_data)

            # ── Tool call finished ────────────────────────────────────────────
            elif kind == "on_tool_end":
                step = tool_steps.pop(name, None)
                if step:
                    output = event.get("data", {}).get("output", "")
                    step.output = str(output)[:500] + ("…" if len(str(output)) > 500 else "")
                    await step.__aexit__(None, None, None)

            # ── Graph state update ────────────────────────────────────────────
            elif kind == "on_chain_end" and name == "LangGraph":
                new_state = event["data"].get("output", new_state)

    except Exception as e:
        await msg_placeholder.update()
        await cl.Message(content=f"An error occurred: {str(e)}\n\nPlease try again.").send()
        return

    # Close any unclosed steps
    for step in tool_steps.values():
        await step.__aexit__(None, None, None)

    # Finalise the streamed message
    if not final_text:
        # Non-streaming response (confirmation flows)
        last = new_state["messages"][-1] if new_state.get("messages") else None
        if isinstance(last, AIMessage) and isinstance(last.content, str):
            final_text = last.content
        msg_placeholder.content = final_text

    await msg_placeholder.update()

    # Show sources panel if available
    if new_state.get("sources_used"):
        sources = new_state["sources_used"]
        source_lines = [
            f"- **{s['source_file']}** (authority {s['authority_level']}, page {s.get('page_num','?')}): {s.get('preview','')[:80]}…"
            for s in sources[:5]
        ]
        await cl.Message(
            content="**Sources consulted:**\n" + "\n".join(source_lines),
            author="Sources",
        ).send()

    # Conflict warning
    if new_state.get("conflict_detected"):
        await cl.Message(
            content="⚠️ **Conflict detected** between sources. The higher-authority source (customer agreement) was applied.",
            author="Warning",
        ).send()

    # Persist updated state
    cl.user_session.set("agent_state", new_state)
