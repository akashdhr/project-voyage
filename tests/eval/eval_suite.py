import json
from dataclasses import dataclass
from typing import List, Dict, Any
from app.domain.models import FlightSearchRequest

@dataclass
class EvalTestCase:
    id: str
    query: str
    expected_hard_constraints: Dict[str, Any]
    description: str

GOLDEN_DATASET = [
    EvalTestCase(
        id="basic_cheapest",
        query="Find me the cheapest flight from Bangalore to London in October. Max 1 stop, checked baggage required.",
        expected_hard_constraints={
            "origin": "BLR",
            "destination": "LHR", # Or LON
            "max_stops": 1,
            "baggage_required": True
        },
        description="Basic constraints with max stops and baggage"
    ),
    EvalTestCase(
        id="flexible_dates",
        query="Flights from SFO to JFK, sometime next week. I'm very flexible with dates.",
        expected_hard_constraints={
            "origin": "SFO",
            "destination": "JFK"
        },
        description="Flexible dates check"
    ),
    EvalTestCase(
        id="nearby_airports",
        query="Cheapest flight from NYC to anywhere near Tokyo.",
        expected_hard_constraints={
            "origin": "NYC",
            "destination": "TYO"
        },
        description="Nearby airports check"
    ),
    EvalTestCase(
        id="best_value",
        query="I want the best value flight from DEL to SFO. I prefer British Airways but will pay slightly more for a faster flight.",
        expected_hard_constraints={
            "origin": "DEL",
            "destination": "SFO"
        },
        description="Best value and airline preference check"
    )
]

def evaluate_parsing(llm_parser_func):
    """
    Evaluates the LLM parsing logic against the golden dataset.
    This can be run offline to measure parsing accuracy without running searches.
    """
    results = []
    for test_case in GOLDEN_DATASET:
        try:
            parsed_request: FlightSearchRequest = llm_parser_func(test_case.query)
            
            # Simple evaluation logic
            success = True
            for k, v in test_case.expected_hard_constraints.items():
                parsed_val = getattr(parsed_request.hard_constraints, k, None)
                if parsed_val != v and str(parsed_val).upper() != str(v).upper():
                    # Check if destination is generically matched (e.g., LHR vs LON)
                    if k == "destination" and v in ["LHR", "LON"] and parsed_val in ["LHR", "LON", "LGW"]:
                        continue
                    success = False
                    break
                    
            results.append({
                "id": test_case.id,
                "success": success,
                "parsed": parsed_request.model_dump()
            })
        except Exception as e:
            results.append({
                "id": test_case.id,
                "success": False,
                "error": str(e)
            })
            
    # Calculate metrics
    success_rate = sum(1 for r in results if r.get("success")) / len(results)
    return {
        "total": len(results),
        "success_rate": success_rate,
        "details": results
    }

if __name__ == "__main__":
    print(f"Golden dataset contains {len(GOLDEN_DATASET)} representative queries.")
