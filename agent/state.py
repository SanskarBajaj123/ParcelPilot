"""
AgentState and UserContext — shared across all LangGraph nodes.
"""

import time
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class UserContext(TypedDict):
    account_id:   Optional[str]                        # "ACCT-001"; None for internal staff
    account_name: Optional[str]                        # "Northstar Logistics"; None for internal
    role:         Literal["customer", "internal"]
    name:         str                                  # display name of the user


class SourceRef(TypedDict):
    source_file:     str
    authority_level: int
    doc_type:        str
    page_num:        int
    preview:         str   # first 120 chars of chunk


class PendingAction(TypedDict):
    action_type: str        # escalate_ticket | update_ticket_status | create_followup_task
    payload:     dict
    summary:     str        # human-readable description shown to user for confirmation


class AgentState(TypedDict):
    # ── Core ──────────────────────────────────────────────────────────────────
    messages:              Annotated[list, add_messages]   # full history; LangGraph reducer
    message_timestamps:    list[float]                     # parallel Unix timestamps per message
    user_context:          UserContext                     # immutable per session

    # ── Confirmation gate ──────────────────────────────────────────────────────
    confirmation_state:    Literal["IDLE", "PENDING_CONFIRMATION"]
    pending_action:        Optional[PendingAction]

    # ── Transparency (rendered in Chainlit tool panel) ─────────────────────────
    sources_used:          list[SourceRef]
    conflict_detected:     bool
    tool_calls_this_turn:  list[str]


def make_initial_state(user_context: UserContext) -> AgentState:
    return AgentState(
        messages=[],
        message_timestamps=[],
        user_context=user_context,
        confirmation_state="IDLE",
        pending_action=None,
        sources_used=[],
        conflict_detected=False,
        tool_calls_this_turn=[],
    )
