from typing import List, Optional
from app.domain.models import Flight, FlightSearchRequest

def filter_candidates(candidates: List[Flight], request: FlightSearchRequest) -> List[Flight]:
    """Filters flights strictly based on hard constraints."""
    hard = request.hard_constraints
    valid = []
    
    for c in candidates:
        # Check max stops
        if hard.max_stops is not None and c.stops is not None:
            if c.stops > hard.max_stops:
                continue
                
        # Check baggage requirement
        if hard.baggage_required:
            if c.baggage_included is False:
                continue # Only reject if explicitly false, None might mean unknown
                
        # In a real app, we'd also check precise date matches and exact airport matches here
        # For demo purposes, we trust the search query largely handled dates/airports
        
        valid.append(c)
        
    return valid

def deduplicate_candidates(candidates: List[Flight]) -> List[Flight]:
    """Removes duplicate flights based on airline, flight number, and price."""
    seen = set()
    deduped = []
    for f in candidates:
        # Create a unique key
        key = f"{f.airline}-{f.flight_number}-{f.price}-{f.stops}"
        if key not in seen:
            seen.add(key)
            deduped.append(f)
    return deduped

def rank_candidates(candidates: List[Flight], request: FlightSearchRequest) -> List[Flight]:
    """Ranks candidates based on the requested optimization goal."""
    if not candidates:
        return []
        
    goal = request.soft_preferences.optimization_goal.lower()
    
    # We only rank flights with a known price for cheapest/best_value
    priced = [c for c in candidates if c.price is not None]
    unpriced = [c for c in candidates if c.price is None]
    
    if goal == "fastest":
        priced.sort(key=lambda x: x.duration_minutes if x.duration_minutes else 9999)
    elif goal == "best_value":
        # Simple scoring: price + penalty for duration and stops
        def value_score(f: Flight) -> float:
            score = f.price
            if f.duration_minutes:
                # Add 100 currency units penalty per hour over 5 hours (arbitrary for demo)
                extra_hours = max(0, (f.duration_minutes / 60) - 5)
                score += extra_hours * 100
            if f.stops:
                # Add 500 currency units penalty per stop
                score += f.stops * 500
            return score
            
        priced.sort(key=value_score)
    else:
        # Default to cheapest
        priced.sort(key=lambda x: x.price)
        
    return priced + unpriced

def evaluate_evidence(candidates: List[Flight]) -> None:
    """Evaluates evidence scores for candidates (in-place).
    Since it's a property now, this is a placeholder if we need more complex logic.
    """
    pass
