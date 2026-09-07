from langgraph.graph import StateGraph, START, END

from app.agent.state import VoyageState
from app.agent.nodes import (
    parse_request_node,
    search_node,
    tool_node,
    extract_candidates_node,
    evaluate_node,
    evaluate_routing,
    rank_node,
    recommend_node,
    should_continue
)

def build_graph():
    """Constructs the LangGraph for the Voyage agent."""
    workflow = StateGraph(VoyageState)
    
    # Add nodes
    workflow.add_node("parse_request", parse_request_node)
    workflow.add_node("search", search_node)
    workflow.add_node("execute_tools", tool_node)
    workflow.add_node("extract_candidates", extract_candidates_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("rank", rank_node)
    workflow.add_node("recommend", recommend_node)
    
    # Define edges
    workflow.add_edge(START, "parse_request")
    workflow.add_edge("parse_request", "search")
    
    # Conditional edge after search: did the LLM call a tool?
    workflow.add_conditional_edges("search", should_continue, {
        "execute_tools": "execute_tools",
        "extract_candidates": "extract_candidates"
    })
    
    # Tools flow into extraction
    workflow.add_edge("execute_tools", "extract_candidates")
    
    # Extraction flows into evaluation
    workflow.add_edge("extract_candidates", "evaluate")
    
    # Evaluation routes to search or rank
    workflow.add_conditional_edges("evaluate", evaluate_routing, {
        "search": "search",
        "rank": "rank"
    })
    
    workflow.add_edge("rank", "recommend")
    workflow.add_edge("recommend", END)
    
    return workflow.compile()
