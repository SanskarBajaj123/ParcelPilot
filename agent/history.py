"""
Time-based conversation history filtering.

Strategy:
  1. Keep all messages from the last HISTORY_WINDOW_HOURS hours.
  2. Never exceed MAX_HISTORY_MESSAGES total messages.
  3. Never break a tool-call pair (AIMessage-with-tool_calls + ToolMessage must stay together).
  4. If the window would include zero messages, keep the last 2 pairs as a minimum
     so the model always has some context.
"""

import os
import time
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage


HISTORY_WINDOW_HOURS  = float(os.getenv("HISTORY_WINDOW_HOURS", "1"))
MAX_HISTORY_MESSAGES  = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))


def stamp(msg: BaseMessage) -> BaseMessage:
    """Attach a Unix timestamp to a message's additional_kwargs."""
    msg.additional_kwargs["_ts"] = time.time()
    return msg


def get_ts(msg: BaseMessage) -> float:
    return msg.additional_kwargs.get("_ts", 0.0)


def filter_history(
    messages: list[BaseMessage],
    window_hours: float = HISTORY_WINDOW_HOURS,
    max_messages: int   = MAX_HISTORY_MESSAGES,
) -> list[BaseMessage]:
    """
    Return the slice of messages that fall within the time window
    AND within the max_messages cap — whichever is more restrictive.
    Tool-call pairs are kept intact.
    """
    if not messages:
        return []

    cutoff = time.time() - (window_hours * 3600)

    # --- Step 1: time filter ---
    time_filtered = [m for m in messages if get_ts(m) >= cutoff]

    # --- Step 2: message count cap (keep most recent) ---
    if len(time_filtered) > max_messages:
        time_filtered = time_filtered[-max_messages:]

    # --- Step 3: ensure we have at least the last 4 messages (2 pairs) ---
    if len(time_filtered) < 4 and len(messages) >= 4:
        time_filtered = messages[-4:]

    # --- Step 4: repair broken tool-call pairs at the start ---
    time_filtered = _repair_tool_pairs(time_filtered, messages)

    return time_filtered


def _repair_tool_pairs(
    filtered: list[BaseMessage],
    full_history: list[BaseMessage],
) -> list[BaseMessage]:
    """
    If the first message in `filtered` is a ToolMessage, its corresponding
    AIMessage (with tool_calls) was cut off. Walk back into full_history to
    find and prepend it.
    """
    if not filtered:
        return filtered

    first = filtered[0]
    if not isinstance(first, ToolMessage):
        return filtered

    # Find the AIMessage that issued this tool call in full_history
    tool_call_id = first.tool_call_id
    for msg in reversed(full_history):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            if any(tc["id"] == tool_call_id for tc in msg.tool_calls):
                return [msg] + filtered

    # Could not find it (shouldn't happen in normal operation)
    return filtered


def summarise_prompt(messages: list[BaseMessage]) -> str:
    """
    Produce a plain-text summary prompt for messages that are being dropped
    (used by the optional summarisation node in long internal-ops sessions).
    """
    lines = ["Summarise this portion of a support conversation. Preserve: decisions made, data looked up, actions taken, open issues."]
    for m in messages:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)
