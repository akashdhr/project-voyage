import operator
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.domain.models import Flight, FlightSearchRequest

class VoyageState(TypedDict):
    # LangGraph state needs to track messages for tool calling
    messages: Annotated[List[BaseMessage], operator.add]
    
    # The parsed user request
    request: Optional[FlightSearchRequest]
    
    # Track the number of search iterations
    iteration_count: int
    
    # Extracted candidate flights
    candidate_flights: List[Flight]
    
    # The final natural language recommendation
    recommendation: Optional[str]
    
    # Rationale from the evaluator (reasoning for next step or stopping)
    search_rationale: Optional[str]
    
    # Feedback from the evaluator for the next search iteration
    search_feedback: Optional[str]
    
    # Any errors that occurred
    error: Optional[str]
