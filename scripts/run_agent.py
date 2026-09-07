import asyncio
import os
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from app.agent.graph import build_graph
from app.observability.logger import log_agent_event
from app.domain.models import AgentEventType

async def run_query():
    load_dotenv()
    
    # Ensure tracing is disabled explicitly as requested
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    graph = build_graph()
    query = "Find me the cheapest flight from Bangalore to London in October. I am flexible with dates and nearby airports. I need 1 stop max."
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "iteration_count": 0,
        "candidate_flights": []
    }
    
    log_agent_event(
        AgentEventType.GRAPH_STARTED, 
        f"Starting graph execution for query: {query}",
        metadata={"query": query}
    )
    
    # We use LangGraph's native event streaming (astream_events v2) to get rich internal details 
    # without needing an external tracing platform like LangSmith.
    async for event in graph.astream_events(initial_state, version="v2"):
        kind = event["event"]
        name = event["name"]
        
        # 1. Track Node Executions
        # Our LangGraph nodes are run as standard chains internally
        if kind == "on_chain_start" and name in ["parse_request", "search", "extract_candidates", "evaluate", "rank", "recommend"]:
            log_agent_event(AgentEventType.NODE_STARTED, f"Node {name} started", node=name)
            
        elif kind == "on_chain_end" and name in ["parse_request", "search", "extract_candidates", "evaluate", "rank", "recommend"]:
            # Capture specific Voyage-level state updates emitted by the node
            data = event.get("data", {}).get("output", {})
            meta = {}
            if isinstance(data, dict):
                # Did the evaluator request a search expansion?
                if "search_feedback" in data and data["search_feedback"]:
                    meta["search_feedback"] = data["search_feedback"]
                    log_agent_event(
                        AgentEventType.SEARCH_EXPANDED, 
                        f"Search expansion feedback: {data['search_feedback']}", 
                        node=name, 
                        metadata=meta
                    )
                # Did we update the candidates list?
                if "candidate_flights" in data:
                    meta["flight_count"] = len(data["candidate_flights"])
                    log_agent_event(
                        AgentEventType.CANDIDATES_UPDATED, 
                        f"Currently tracking {meta['flight_count']} valid flights.", 
                        node=name, 
                        metadata=meta
                    )
                # Did we get the final recommendation?
                if "recommendation" in data:
                    meta["recommendation"] = data["recommendation"]
                    
            log_agent_event(AgentEventType.NODE_COMPLETED, f"Node {name} completed", node=name, metadata=meta)
            
        # 2. Track Tool Executions (Tavily)
        elif kind == "on_tool_start":
            tool_input = event.get("data", {}).get("input")
            log_agent_event(
                AgentEventType.TOOL_STARTED, 
                f"Executing tool: {name}", 
                tool=name, 
                metadata={"input": tool_input}
            )
            
        elif kind == "on_tool_end":
            log_agent_event(AgentEventType.TOOL_COMPLETED, f"Tool {name} completed", tool=name)

    log_agent_event(AgentEventType.GRAPH_COMPLETED, "Graph execution finished")

if __name__ == "__main__":
    asyncio.run(run_query())
