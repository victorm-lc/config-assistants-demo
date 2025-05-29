"""React Agent.

This module defines a custom reasoning and action agent graph.
It invokes tools in a simple loop.
"""

from react_agent.graph import make_graph
from react_agent.graph_without_config import make_graph as make_graph_without_config

async def compiled_graph():
    return make_graph(None, "ReAct Agent Demo")

async def compiled_graph_without_config():
    return make_graph_without_config(None,"ReAct Agent w/o Config Demo")

__all__ = ["compiled_graph", "compiled_graph_without_config"]
