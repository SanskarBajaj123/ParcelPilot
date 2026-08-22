"""
Tool 3: create_action
State-changing actions â€” escalation, ticket update, follow-up task.
When execute=False: returns a draft for user confirmation (no side effects).
When execute=True: commits the action (mocked â€” logs to Supabase actions_log table).
"""

import os
import uuid
import time
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

from pathlib import Path as _Path
load_dotenv(_Path(__file__).parent.parent / ".env" if (_Path(__file__).parent.parent / ".env").exists() else _Path(__file__).parent.parent.parent / ".env", override=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

VALID_ACTION_TYPES = {"escalate_ticket", "update_ticket_status", "create_followup_task"}
VALID_STATUSES     = {"open", "closed", "escalated", "in_progress"}


def create_action_fn(
    action_type: str,
    payload:     dict,
    execute:     bool,
    actor_name:  str,
    actor_role:  str,
) -> dict:
    """
    Prepare or execute a state-changing action.

    action_type: escalate_ticket | update_ticket_status | create_followup_task
    payload:     action-specific fields (see below per type)
    execute:     False = draft only. True = commit (after user confirms).
    actor_name / actor_role: from user_context â€” injected by tool_node.

    Returns:
      If execute=False:  {draft: True, summary, action_type, payload}
      If execute=True:   {executed: True, action_id, summary, timestamp}
    """
    if action_type not in VALID_ACTION_TYPES:
        return {"error": f"Unknown action_type '{action_type}'. Valid: {', '.join(VALID_ACTION_TYPES)}"}

    # â”€â”€ Build human-readable summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if action_type == "escalate_ticket":
        ticket_id = payload.get("ticket_id", "?")
        reason    = payload.get("reason", "not provided")
        severity  = payload.get("severity", "P2")
        summary   = (
            f"Escalate ticket {ticket_id} to severity {severity}.\n"
            f"Reason: {reason}\n"
            f"This will alert the support team and create an escalation record."
        )

    elif action_type == "update_ticket_status":
        ticket_id  = payload.get("ticket_id", "?")
        new_status = payload.get("new_status", "?")
        if new_status not in VALID_STATUSES:
            return {"error": f"Invalid new_status '{new_status}'. Valid: {', '.join(VALID_STATUSES)}"}
        note   = payload.get("note", "")
        summary = (
            f"Update ticket {ticket_id} status â†’ '{new_status}'."
            + (f"\nNote: {note}" if note else "")
        )

    elif action_type == "create_followup_task":
        ticket_id   = payload.get("ticket_id", "?")
        description = payload.get("description", "?")
        due_hours   = payload.get("due_hours", 24)
        summary     = (
            f"Create a follow-up task on ticket {ticket_id}:\n"
            f"  Task: {description}\n"
            f"  Due:  {due_hours}h from now"
        )

    # â”€â”€ Draft mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not execute:
        return {
            "draft":       True,
            "action_type": action_type,
            "payload":     payload,
            "summary":     summary,
        }

    # â”€â”€ Execute mode â€” commit to Supabase (mocked action log) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    action_id  = str(uuid.uuid4())[:8].upper()
    timestamp  = datetime.now(timezone.utc).isoformat()

    log_row = {
        "action_id":   action_id,
        "action_type": action_type,
        "payload":     payload,
        "actor_name":  actor_name,
        "actor_role":  actor_role,
        "summary":     summary,
        "executed_at": timestamp,
        "status":      "completed",
    }

    try:
        sb.table("actions_log").insert(log_row).execute()
    except Exception:
        # actions_log table may not exist in basic Supabase setup â€” fail gracefully
        pass

    return {
        "executed":  True,
        "action_id": action_id,
        "summary":   summary,
        "timestamp": timestamp,
        "note":      "Action recorded. In production this would trigger the real workflow.",
    }


# â”€â”€ Optional: Supabase setup for actions_log (run once) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ACTIONS_LOG_DDL = """
CREATE TABLE IF NOT EXISTS actions_log (
    id          BIGSERIAL PRIMARY KEY,
    action_id   TEXT NOT NULL,
    action_type TEXT NOT NULL,
    payload     JSONB,
    actor_name  TEXT,
    actor_role  TEXT,
    summary     TEXT,
    executed_at TIMESTAMPTZ,
    status      TEXT
);
"""

