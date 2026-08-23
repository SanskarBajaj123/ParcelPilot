"""
Conversation history filtering for the ParcelPilot agent.

Strategy: keep the last MAX_TURNS complete conversation turns.
A "turn" is one HumanMessage plus all the AI and ToolMessages that follow it
until the next HumanMessage. This guarantees the LLM always sees proper
Human -> AI -> Human -> AI alternation, and never hallucinates from stale
context outside the window.

Default: 3 human turns + their paired AI responses (and tool calls).
"""

import os
import time
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage


MAX_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "3"))


def stamp(msg: BaseMessage) -> BaseMessage:
    """Attach a Unix timestamp to a message's additional_kwargs."""
    msg.additional_kwargs["_ts"] = time.time()
    return msg


def get_ts(msg: BaseMessage) -> float:
    return msg.additional_kwargs.get("_ts", 0.0)


def filter_history(
    messages: list[BaseMessage],
    max_turns: int = MAX_TURNS,
    # Legacy params accepted but ignored - turn-based is strictly better
    window_hours: float = 1.0,
    max_messages: int = 30,
) -> list[BaseMessage]:
    """
    Return the last `max_turns` complete conversation turns.

    A turn starts at each HumanMessage and includes all subsequent AI and
    ToolMessages until the next HumanMessage. Tool-call pairs are always
    kept intact within their turn.

    With max_turns=3 this gives the LLM exactly 3 user messages and up to
    3 final assistant responses (plus any intermediate tool messages) -
    enough context to avoid repetition while preventing hallucination from
    stale prior turns.
    """
    if not messages:
        return []

    # Split the full history into turns, each starting at a HumanMessage
    turns: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, HumanMessage) and current:
            turns.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        turns.append(current)

    # Keep only the most recent max_turns turns
    selected = turns[-max_turns:]

    return [msg for turn in selected for msg in turn]


def summarise_prompt(messages: list[BaseMessage]) -> str:
    """
    Produce a plain-text summary of messages (for debug / logging).
    """
    lines = [
        "Summarise this portion of a support conversation. "
        "Preserve: decisions made, data looked up, actions taken, open issues."
    ]
    for m in messages:
        role = type(m).__name__.replace("Message", "")
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:300]}")
    return "\n".join(lines)
