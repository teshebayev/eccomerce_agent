from operator import add
from typing import Annotated, Any, Dict, List

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Prefetch, Document, FusionQuery


from api.agents.export_langgraph_png import save_langgraph_visualization

from api.agents.agents_ollama import (
    RAGUsedContext,
    ToolCall,
    agent_node,
    intent_router_node,
)
from api.agents.tools import get_formatted_context
from api.agents.utils.utils import get_tool_descriptions


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 1
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []


def tool_router(state: State) -> str:
    """Decide whether to continue or end"""
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def intent_router_conditional_edges(state: State):
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"


# === Workflow ===
workflow = StateGraph(State)

tools = [get_formatted_context]
tool_node = ToolNode(tools)
tool_descriptions = get_tool_descriptions(tools)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "agent_node",
    tool_router,
    {
        "tools": "tool_node",
        "end": END,
    },
)

workflow.add_edge("tool_node", "agent_node")

graph = workflow.compile()

# path = save_langgraph_visualization(graph)
# print(f"LangGraph visualization saved to: {path}")

# === Agent Execution Function ===
def run_agent(question: str) -> dict:
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tool_descriptions,
    }

    result = graph.invoke(initial_state)

    return result

def rag_agent_wrapper(question: str):
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result = run_agent(question)

    used_context = []

    for item in result.get("references", []):
        points, _ = qdrant_client.scroll(
            collection_name="Amazon_items_collection1_hybrid_search",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.id),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            continue

        payload = points[0].payload or {}

        used_context.append(
            {
                "image_url": payload.get("image"),
                "price": payload.get("price"),
                "description": item.description,
            }
        )

    return {
        "answer": result.get("answer", ""),
        "used_context": used_context,
    }
