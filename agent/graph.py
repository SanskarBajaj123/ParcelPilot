"""
LangGraph StateGraph definition for ParcelPilot Support Agent.

Flow:
  START -> agent_node
    -> if tool_calls: tool_node -> agent_node  (loop until no tool calls)
    -> else: END
"""

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

from .state import AgentState
from .nodes import agent_node, tool_node


def route_after_agent(state: AgentState) -> str:
    """If the AI produced tool calls, dispatch them; otherwise the turn is done."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_node"
    return END


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("agent_node", agent_node)
    builder.add_node("tool_node",  tool_node)

    builder.add_edge(START, "agent_node")
    builder.add_conditional_edges("agent_node", route_after_agent, {
        "tool_node": "tool_node",
        END:         END,
    })
    builder.add_edge("tool_node", "agent_node")

    return builder.compile(name="ParcelPilot Support Agent")


# Singleton compiled graph
graph = build_graph()
