"""
LangGraph StateGraph definition for ParcelPilot Support Agent.

Flow:
  START → agent_node
    → if tool_calls:    tool_node → agent_node  (loop until no tool calls)
    → if PENDING_CONFIRMATION + human msg: confirm_node → agent_node
    → else: END
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage

from .state import AgentState
from .nodes import agent_node, tool_node, confirm_node


# ── Edge functions ────────────────────────────────────────────────────────────

def route_after_agent(state: AgentState) -> str:
    """
    After agent_node responds:
      - If the AI produced tool calls → go to tool_node.
      - Otherwise → END (response is ready for the user).
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_node"
    return END


def route_after_tool(state: AgentState) -> str:
    """
    After tool_node completes:
      - Always loop back to agent_node so the LLM can process tool results.
    """
    return "agent_node"


def route_entry(state: AgentState) -> str:
    """
    Entry routing: if we're in PENDING_CONFIRMATION and the last message
    is a HumanMessage, go to confirm_node instead of agent_node.
    """
    if (
        state.get("confirmation_state") == "PENDING_CONFIRMATION"
        and state["messages"]
        and isinstance(state["messages"][-1], HumanMessage)
    ):
        return "confirm_node"
    return "agent_node"


# ── Graph definition ──────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("agent_node",   agent_node)
    builder.add_node("tool_node",    tool_node)
    builder.add_node("confirm_node", confirm_node)

    # Entry point: route to either agent_node or confirm_node
    builder.add_conditional_edges(START, route_entry, {
        "agent_node":   "agent_node",
        "confirm_node": "confirm_node",
    })

    # After agent: check for tool calls
    builder.add_conditional_edges("agent_node", route_after_agent, {
        "tool_node": "tool_node",
        END:         END,
    })

    # After tools: always go back to agent
    builder.add_edge("tool_node", "agent_node")

    # After confirm: back to agent for a response
    builder.add_edge("confirm_node", "agent_node")

    return builder.compile()


# Singleton compiled graph
graph = build_graph()
