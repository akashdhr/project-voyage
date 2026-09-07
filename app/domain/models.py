from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

class FlightSearchRequest(BaseModel):
    origin: str = Field(description="Origin city or airport code")
    destination: str = Field(description="Destination city or airport code")
    departure_start: Optional[str] = Field(None, description="Earliest departure date (YYYY-MM-DD)")
    departure_end: Optional[str] = Field(None, description="Latest departure date (YYYY-MM-DD)")
    return_start: Optional[str] = Field(None, description="Earliest return date (YYYY-MM-DD)")
    return_end: Optional[str] = Field(None, description="Latest return date (YYYY-MM-DD)")
    flexible_dates: bool = Field(False, description="Whether the user is flexible with dates")
    flexible_origin_airports: bool = Field(False, description="Whether to consider nearby origin airports")
    flexible_destination_airports: bool = Field(False, description="Whether to consider nearby destination airports")
    cabin: Optional[str] = Field(None, description="Cabin class (economy, premium_economy, business, first)")
    max_stops: Optional[int] = Field(None, description="Maximum number of stops allowed")
    baggage_required: bool = Field(False, description="Whether checked baggage is explicitly required")
    optimization_goal: str = Field("cheapest", description="What to optimize for: cheapest, best_value, shortest_duration, fewest_stops")

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"

class Flight(BaseModel):
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    origin_airport: Optional[str] = None
    destination_airport: Optional[str] = None
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    stops: Optional[int] = 0
    price: Optional[float] = None
    currency: Optional[str] = None
    baggage_included: Optional[bool] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED

class AgentEventType(str, Enum):
    GRAPH_STARTED = "GRAPH_STARTED"
    NODE_STARTED = "NODE_STARTED"
    NODE_COMPLETED = "NODE_COMPLETED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    SEARCH_EXPANDED = "SEARCH_EXPANDED"
    CANDIDATES_UPDATED = "CANDIDATES_UPDATED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"
    AGENT_ERROR = "AGENT_ERROR"
    GRAPH_COMPLETED = "GRAPH_COMPLETED"

class AgentEvent(BaseModel):
    event_type: AgentEventType
    timestamp: datetime = Field(default_factory=datetime.now)
    node: Optional[str] = None
    tool: Optional[str] = None
    status: Optional[str] = None
    message: str
    metadata: Optional[Dict[str, Any]] = None
