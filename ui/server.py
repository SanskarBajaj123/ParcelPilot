"""
ParcelPilot - FastAPI + WebSocket UI server.

Replaces Chainlit entirely. Run:
  uvicorn ui.server:app --port 8000 --reload
  (from inside parcelpilot-agent/)
"""

import os, time, hashlib, asyncio, json, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from supabase import create_client
from langchain_core.messages import HumanMessage, AIMessage

_env = Path(__file__).parent.parent / ".env"
load_dotenv(_env if _env.exists() else _env.parent.parent / ".env", override=True)

from agent.graph import graph
from agent.state import AgentState, UserContext, make_initial_state
from agent.history import stamp
from agent.tools.create_action import create_action_fn
from proactive.detector import detect_issues

log = logging.getLogger("parcelpilot.server")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SVC = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
sb_svc = create_client(SUPABASE_URL, SUPABASE_SVC)

MAX_ATTEMPTS = 3

_CUSTOMER_PINS: dict[str, str] = {
    "ACCT-001": "1234",
    "ACCT-002": "5678",
    "ACCT-003": "4321",
    "ACCT-004": "8765",
}

def _h(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

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

# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="ParcelPilot Agent")
STATIC = Path(__file__).parent / "static"

@app.get("/")
async def root():
    return FileResponse(STATIC / "index.html")

app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ── Helpers ─────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

async def _send(ws: WebSocket, **kwargs) -> None:
    try:
        await ws.send_json(kwargs)
    except Exception:
        pass

# ── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session: dict[str, Any] = {
        "auth_verified": False,
        "auth_attempts": 0,
        "user_context": None,
        "state": None,
        "log": [],
    }

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            t   = msg.get("type", "")

            # ── Customer auth ──────────────────────────────────────────────
            if t == "auth_customer":
                raw_id = msg.get("account_id", "").strip().upper()
                pin    = msg.get("pin", "").strip()

                rows = sb_svc.table("accounts") \
                    .select("account_id,account_name,plan") \
                    .eq("account_id", raw_id).execute().data

                if not rows:
                    await _send(websocket, type="auth_error",
                                message=f"Account {raw_id} not found. Check your Account ID.")
                    continue

                acct     = rows[0]
                expected = _CUSTOMER_PINS.get(acct["account_id"])
                if not expected or pin != expected:
                    session["auth_attempts"] += 1
                    left = MAX_ATTEMPTS - session["auth_attempts"]
                    if left <= 0:
                        await _send(websocket, type="auth_locked",
                                    message="Too many failed attempts. Please refresh.")
                    else:
                        await _send(websocket, type="auth_error",
                                    message=f"Incorrect PIN. {left} attempt(s) remaining.")
                    continue

                ctx: UserContext = {
                    "account_id":   acct["account_id"],
                    "account_name": acct["account_name"],
                    "role":         "customer",
                    "name":         acct["account_name"],
                }
                session.update(auth_verified=True, user_context=ctx,
                               state=make_initial_state(ctx))
                session["log"].append({
                    "time": _ts(), "type": "AUTH",
                    "detail": f"CUSTOMER_OK | {acct['account_id']} - {acct['account_name']}",
                })
                await _send(websocket, type="auth_ok", role="customer",
                            account_id=acct["account_id"],
                            name=acct["account_name"],
                            plan=acct.get("plan",""))

            # ── Staff auth ─────────────────────────────────────────────────
            elif t == "auth_staff":
                username = msg.get("username", "").strip().lower()
                password = msg.get("password", "")

                if username not in _STAFF_CREDS or _STAFF_CREDS[username] != _h(password):
                    session["auth_attempts"] += 1
                    left = MAX_ATTEMPTS - session["auth_attempts"]
                    if left <= 0:
                        await _send(websocket, type="auth_locked", message="Too many failed attempts.")
                    else:
                        await _send(websocket, type="auth_error",
                                    message=f"Unknown username or incorrect password. {left} attempt(s) remaining.")
                    continue

                name = _STAFF_NAMES[username]
                ctx = {
                    "account_id":   None,
                    "account_name": None,
                    "role":         "internal",
                    "name":         name,
                }
                session.update(auth_verified=True, user_context=ctx,
                               state=make_initial_state(ctx))
                session["log"].append({
                    "time": _ts(), "type": "AUTH",
                    "detail": f"STAFF_OK | {username} ({name})",
                })
                await _send(websocket, type="auth_ok", role="internal", name=name)
                # Fire proactive scan in background
                asyncio.create_task(_run_scan(websocket, session))

            # ── Chat message ───────────────────────────────────────────────
            elif t == "message":
                if not session["auth_verified"]:
                    await _send(websocket, type="error", message="Not authenticated.")
                    continue
                content = msg.get("content", "").strip()
                if not content:
                    continue
                session["log"].append({
                    "time": _ts(), "type": "USER_MSG",
                    "detail": content[:120],
                })
                asyncio.create_task(_handle_chat(websocket, session, content))

            # ── Confirmation response ──────────────────────────────────────
            elif t == "confirm":
                if not session["auth_verified"]:
                    continue
                confirmed = bool(msg.get("value"))
                session["log"].append({
                    "time": _ts(), "type": "USER_MSG",
                    "detail": f"Confirmation button: {'YES' if confirmed else 'NO'}",
                })
                asyncio.create_task(_handle_confirm(websocket, session, confirmed))

            # ── Get session logs ───────────────────────────────────────────
            elif t == "get_logs":
                if not session["auth_verified"]:
                    await _send(websocket, type="error", message="Not authenticated.")
                    continue
                await _send(websocket, type="logs_data", events=session["log"])

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception:
        log.exception("WebSocket error")


async def _run_scan(ws: WebSocket, session: dict) -> None:
    """Run proactive scan in thread pool and send results."""
    t0 = time.time()
    try:
        results = await asyncio.get_event_loop().run_in_executor(None, detect_issues)
        lat = int((time.time() - t0) * 1000)
        session["log"].append({
            "time": _ts(), "type": "PROACTIVE_SCAN",
            "detail": f"Scan complete | {lat}ms",
            "latency_ms": lat,
        })
        await _send(ws, type="proactive_scan", results=results, latency_ms=lat)
    except Exception as e:
        log.exception("Proactive scan failed")
        await _send(ws, type="proactive_scan",
                    results={"error": str(e)}, latency_ms=0)


CONFIRM_TIMEOUT_SECS = 300   # 5 minutes


async def _handle_confirm(ws: WebSocket, session: dict, confirmed: bool) -> None:
    """
    Handle Yes/No button clicks directly - bypasses LangGraph entirely.
    Creates or cancels the pending action and streams a response.
    """
    cs = session["state"].get("confirmation_state", "IDLE")
    pa = session["state"].get("pending_action")
    pa_at = session.get("pending_action_at", 0)

    # Reset confirmation state unconditionally so it doesn't linger
    session["state"]["confirmation_state"] = "IDLE"
    session["state"]["pending_action"] = None
    session.pop("pending_action_at", None)

    if cs != "PENDING_CONFIRMATION" or not pa:
        await _send(ws, type="token",
                    content="No action is currently pending. Please make a new request.")
        await _send(ws, type="message_end", sources=[], conflict=False, requires_confirm=False)
        return

    # 5-minute expiry
    if time.time() - pa_at > CONFIRM_TIMEOUT_SECS:
        await _send(ws, type="token",
                    content="The pending action has expired (5-minute limit). Please request the action again.")
        await _send(ws, type="message_end", sources=[], conflict=False, requires_confirm=False)
        session["log"].append({"time": _ts(), "type": "ACTION_EXPIRED", "detail": pa.get("action_type", "?")})
        return

    if not confirmed:
        await _send(ws, type="token",
                    content="Action cancelled. Let me know if there's anything else I can help with.")
        await _send(ws, type="message_end", sources=[], conflict=False, requires_confirm=False)
        session["log"].append({"time": _ts(), "type": "ACTION_CANCELLED", "detail": pa.get("action_type", "?")})
        # Append cancel exchange to history so context is preserved
        session["state"]["messages"].append(stamp(AIMessage(
            content="Action cancelled. Let me know if there's anything else I can help with."
        )))
        return

    # Execute the action
    user_ctx = session["state"]["user_context"]
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: create_action_fn(
                action_type=pa["action_type"],
                payload=pa["payload"],
                execute=True,
                actor_name=user_ctx.get("name", "unknown"),
                actor_role=user_ctx["role"],
            )
        )
        summary    = result.get("summary", "Action completed.")
        action_id  = result.get("action_id", "N/A")
        ts_str     = result.get("timestamp", "")
        reply_text = f"Done. {summary}\n\nAction ID: **{action_id}**"
        if ts_str:
            reply_text += f"  |  {ts_str}"

        await _send(ws, type="token", content=reply_text)
        session["log"].append({
            "time": _ts(), "type": "ACTION_EXECUTED",
            "detail": f"{pa['action_type']} | {action_id}",
        })
        session["state"]["messages"].append(stamp(AIMessage(content=reply_text)))
    except Exception as e:
        log.exception("Action execution error")
        err_text = f"Failed to execute action: {e}"
        await _send(ws, type="token", content=err_text)
        session["state"]["messages"].append(stamp(AIMessage(content=err_text)))

    await _send(ws, type="message_end", sources=[], conflict=False, requires_confirm=False)


async def _handle_chat(ws: WebSocket, session: dict, content: str) -> None:
    """Stream LangGraph agent response over WebSocket."""
    state: AgentState = session["state"]

    human = stamp(HumanMessage(content=content))
    state["messages"].append(human)
    state["message_timestamps"].append(time.time())

    final_text          = ""
    captured_sources    = []
    captured_conflict   = False
    tool_start_times: dict[str, float] = {}
    llm_start: float | None            = None
    # Fallback: track create_action(execute=False) calls from LLM tool-call events
    # so we can set pending_action even if on_chain_end doesn't carry it.
    _pending_tc_args: dict | None      = None

    try:
        async for event in graph.astream_events(state, version="v2"):
            kind = event["event"]
            data = event.get("data", {})
            name = event.get("name", "")

            # Token streaming
            if kind == "on_chat_model_stream":
                if llm_start is None:
                    llm_start = time.time()
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content"):
                    tok = chunk.content
                    if isinstance(tok, str) and tok:
                        final_text += tok
                        await _send(ws, type="token", content=tok)

            elif kind == "on_chat_model_end":
                if llm_start is not None:
                    lat = int((time.time() - llm_start) * 1000)
                    resp = data.get("output")
                    n_tc = len(getattr(resp, "tool_calls", None) or [])
                    detail = f"LLM: {len(final_text)} chars"
                    if n_tc:
                        detail += f", {n_tc} tool call(s)"
                    session["log"].append({
                        "time": _ts(), "type": "LLM_CALL",
                        "detail": detail, "latency_ms": lat,
                    })
                    llm_start = None
                    # Detect create_action(execute=False) for fallback pending_action capture
                    for tc in (getattr(resp, "tool_calls", None) or []):
                        if tc.get("name") == "create_action":
                            args = tc.get("args") or {}
                            if not args.get("execute", True):
                                _pending_tc_args = args

            elif kind == "on_tool_start":
                run_id = event.get("run_id", name)
                tool_start_times[run_id] = time.time()
                inp    = data.get("input", {})
                detail = json.dumps(inp, default=str)[:140]
                session["log"].append({
                    "time": _ts(), "type": "TOOL_START",
                    "detail": f"{name}: {detail}",
                })
                await _send(ws, type="tool_start", name=name, detail=detail)

            elif kind == "on_tool_end":
                run_id = event.get("run_id", name)
                lat    = int((time.time() - tool_start_times.pop(run_id, time.time())) * 1000)
                out    = data.get("output", "")
                preview = (out.content if hasattr(out, "content") else str(out))[:120]
                session["log"].append({
                    "time": _ts(), "type": "TOOL_END",
                    "detail": f"{name}: done", "latency_ms": lat,
                })
                await _send(ws, type="tool_end", name=name,
                            latency_ms=lat, preview=preview)

            elif kind == "on_chain_end":
                log.info("on_chain_end name=%r keys=%r", name, list(data.get("output", {}).keys()) if isinstance(data.get("output"), dict) else type(data.get("output")).__name__)
                out = data.get("output", {})
                if isinstance(out, dict):
                    srcs = out.get("sources_used") or []
                    if srcs:
                        captured_sources = srcs
                    if out.get("conflict_detected"):
                        captured_conflict = True
                    # Write back state fields that must persist between turns
                    if "confirmation_state" in out:
                        session["state"]["confirmation_state"] = out["confirmation_state"]
                    if "pending_action" in out:
                        session["state"]["pending_action"] = out["pending_action"]
                    # Graph root event has the full final messages list
                    if "messages" in out and name not in ("agent_node", "tool_node", "confirm_node"):
                        session["state"]["messages"] = list(out["messages"])

        session["log"].append({
            "time": _ts(), "type": "AGENT_REPLY",
            "detail": (final_text[:80] + "...") if final_text else "(empty)",
        })

        # Fallback: if on_chain_end didn't capture confirmation_state but the LLM made a
        # create_action(execute=False) call, derive pending_action directly from the tool args.
        if session["state"]["confirmation_state"] != "PENDING_CONFIRMATION" and _pending_tc_args:
            try:
                user_ctx = session["state"]["user_context"]
                draft = create_action_fn(
                    action_type=_pending_tc_args.get("action_type", ""),
                    payload=_pending_tc_args.get("payload", {}),
                    execute=False,
                    actor_name=user_ctx.get("name", "unknown"),
                    actor_role=user_ctx["role"],
                )
                if draft.get("draft"):
                    session["state"]["confirmation_state"] = "PENDING_CONFIRMATION"
                    session["state"]["pending_action"] = {
                        "action_type": draft["action_type"],
                        "payload":     draft["payload"],
                        "summary":     draft["summary"],
                    }
                    log.info("Fallback: set confirmation_state=PENDING_CONFIRMATION from tool args")
            except Exception:
                log.exception("Fallback pending_action derivation failed")

        # Record when the pending action was created so we can expire it after 5 min
        if session["state"]["confirmation_state"] == "PENDING_CONFIRMATION":
            session.setdefault("pending_action_at", time.time())

        # Deduplicate sources by source_file, keep max 5
        seen: set[str] = set()
        deduped: list = []
        for src in captured_sources:
            key = src.get("source_file", str(src)) if isinstance(src, dict) else str(src)
            if key not in seen:
                seen.add(key)
                deduped.append(src)
        requires_confirm = session["state"]["confirmation_state"] == "PENDING_CONFIRMATION"
        await _send(ws, type="message_end",
                    sources=deduped[:5],
                    conflict=captured_conflict,
                    requires_confirm=requires_confirm)

    except Exception as e:
        log.exception("Chat handler error")
        await _send(ws, type="error", message=f"Agent error: {e}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ui.server:app", host="0.0.0.0", port=port, reload=False)
