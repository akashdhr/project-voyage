import json
import os
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from app.domain.models import FlightSearchRequest, Flight
from app.llm.client import get_llm
from app.tools.tavily_search import search_flights_tavily
from app.agent.state import VoyageState

def scrub_pii(text: str) -> str:
    """Basic local scrubber to prevent leaking obvious PII to external APIs."""
    # Redact emails
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
    # Redact obvious phone numbers (simple pattern for demo)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', text)
    return text

def parse_request_node(state: VoyageState) -> VoyageState:
    """Parses the initial user request into a structured search specification."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(FlightSearchRequest)
    
    raw_msg = state["messages"][-1].content
    scrubbed_msg = scrub_pii(raw_msg)
    
    prompt = f"""
    You are a flight search expert. Extract the search constraints from the user's request.
    Request: {scrubbed_msg}
    """
    request_spec = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {"request": request_spec}

def search_node(state: VoyageState) -> VoyageState:
    """Plans and executes the search. Uses feedback from previous iterations to improve queries."""
    llm = get_llm()
    tools = [search_flights_tavily]
    llm_with_tools = llm.bind_tools(tools)
    
    req = state.get("request")
    if not req:
        return {"error": "No request specification found."}
        
    iteration = state.get("iteration_count", 0) + 1
    feedback = state.get("search_feedback")
    
    feedback_str = f"CRITICAL FEEDBACK FROM PREVIOUS SEARCH:\n{feedback}\nAdjust your search queries based on this feedback!" if feedback else ""
    
    sys_prompt = f"""
    You are an agentic flight search researcher. 
    You have access to a web search tool. You may call the tool MULTIPLE TIMES in parallel to try different dates, airports, or search terms.
    
    Constraints:
    Origin: {req.hard_constraints.origin}
    Destination: {req.hard_constraints.destination}
    Flexible dates: {req.flexible_dates}
    Flexible airports: {req.flexible_origin_airports} or {req.flexible_destination_airports}
    
    {feedback_str}
    
    Formulate precise search queries. If the user is flexible, consider searching surrounding airports (e.g. LGW instead of LHR).
    Use the search_flights_tavily tool.
    """
    
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response], "iteration_count": iteration}

# The ToolNode takes care of executing the tool if the LLM called it
tool_node = ToolNode(tools=[search_flights_tavily])

def extract_candidates_node(state: VoyageState) -> VoyageState:
    """Extracts flights from the messy search results into our canonical schema."""
    llm = get_llm()
    
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if not tool_msgs:
        return {} 
        
    # Combine all recent tool messages from the last parallel execution
    last_tool_msgs_content = "\n\n".join([m.content for m in tool_msgs[-5:]])
    
    class FlightExtraction(BaseModel):
        flights: list[Flight]
        
    structured_llm = llm.with_structured_output(FlightExtraction)
    
    prompt = f"""
    Extract a maximum of the TOP 2 best flights mentioned in these search results.
    Keep the extraction extremely brief and fast.
    If the results indicate an error or contain no flights, simply return an empty list.
    Do NOT hallucinate missing data.
    
    Search Results:
    {last_tool_msgs_content}
    """
    
    try:
        extraction = structured_llm.invoke([HumanMessage(content=prompt)])
        current_candidates = state.get("candidate_flights", [])
        
        if extraction and extraction.flights:
            current_candidates.extend(extraction.flights)
            
        return {"candidate_flights": current_candidates}
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}

def evaluate_node(state: VoyageState) -> VoyageState:
    """Evaluates the quality of results and explicitly plans the next search if needed."""
    req = state["request"]
    candidates = state.get("candidate_flights", [])
    iteration = state.get("iteration_count", 0)
    max_iters = int(os.getenv("MAX_SEARCH_ITERATIONS", "3"))
    
    if iteration >= max_iters:
        rationale = f"Decision:\nStop searching\nReason:\nMaximum search iterations ({max_iters}) reached."
        return {"search_feedback": None, "search_rationale": rationale}
        
    llm = get_llm()
    
    class EvaluationDecision(BaseModel):
        sufficient_info: bool
        reasoning: str = Field(..., description="Explain why you are stopping or searching again (e.g. 'Only 2 candidates satisfy hard constraints').")
        action: str = Field(..., description="What the next search will do (e.g. 'Expand destination airport search'). Leave empty if stopping.")
        
    structured_llm = llm.with_structured_output(EvaluationDecision)
    
    prompt = f"""
    User wanted flights from {req.hard_constraints.origin} to {req.hard_constraints.destination}.
    We have found {len(candidates)} flights across {iteration} iterations.
    
    If we have 0 flights, or very few, and the user is flexible, return sufficient_info=false and state your reasoning and action.
    If we have a good selection of flights, return sufficient_info=true and state your reasoning.
    """
    
    try:
        decision = structured_llm.invoke([HumanMessage(content=prompt)])
    except Exception as e:
        # Fallback if LLM fails
        return {"search_feedback": None, "search_rationale": f"Decision:\nStop searching\nReason:\nEvaluation failed: {str(e)}"}
    
    if decision.sufficient_info:
        rationale = f"Decision:\nStop searching\nReason:\n{decision.reasoning}"
        return {"search_feedback": None, "search_rationale": rationale}
    else:
        rationale = f"Decision:\nSearch again\nReason:\n{decision.reasoning}\nAction:\n{decision.action}"
        return {"search_feedback": decision.action, "search_rationale": rationale}

def should_continue(state: VoyageState):
    """Conditional edge after search_node."""
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "execute_tools"
    return "extract_candidates"

def evaluate_routing(state: VoyageState):
    """Routing after evaluate_node based on the presence of search_feedback."""
    iteration = state.get("iteration_count", 0)
    max_iters = int(os.getenv("MAX_SEARCH_ITERATIONS", "3"))
    
    if iteration >= max_iters:
        return "rank"
    
    # If the LLM explicitly provided feedback, we search again
    if state.get("search_feedback"):
        return "search"
        
    return "rank"

from app.domain.logic import filter_candidates, deduplicate_candidates, rank_candidates

def rank_node(state: VoyageState) -> VoyageState:
    """Deterministically filters and ranks the candidates using pure Python."""
    req = state["request"]
    candidates = state.get("candidate_flights", [])
    
    # 1. Filter hard constraints
    valid_candidates = filter_candidates(candidates, req)
    
    # 2. Deduplicate
    deduped = deduplicate_candidates(valid_candidates)
    
    # 3. Rank based on soft preferences
    ranked = rank_candidates(deduped, req)
    
    return {"candidate_flights": ranked}

def recommend_node(state: VoyageState) -> VoyageState:
    """Generates the final natural language recommendation."""
    llm = get_llm()
    candidates = state.get("candidate_flights", [])
    req = state["request"]
    
    if not candidates:
        recommendation = "I could not find any verified flights matching your exact criteria based on web search results."
        return {"recommendation": recommendation}
        
    best = candidates[0]
    
    goal = req.soft_preferences.optimization_goal.upper()
    
    prompt = f"""
    The user asked for: {req}
    
    We ranked {len(candidates)} valid candidates using the {goal} strategy. The best one is:
    Airline: {best.airline}
    Price: {best.price} {best.currency}
    Stops: {best.stops}
    Duration: {best.duration_minutes} mins
    Source: {best.source_url}
    Evidence Score: {best.evidence_score}/5
    Verification: {best.verification_status.value}
    
    Write a concise, friendly recommendation. State why this flight was chosen based on the {goal} strategy.
    Emphasize that these prices are discovered from search engines and are unverified.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"recommendation": response.content}
