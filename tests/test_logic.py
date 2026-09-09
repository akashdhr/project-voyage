import pytest
from app.domain.models import Flight, FlightSearchRequest, HardConstraints, SoftPreferences
from app.domain.logic import filter_candidates, deduplicate_candidates, rank_candidates

@pytest.fixture
def base_request():
    return FlightSearchRequest(
        hard_constraints=HardConstraints(
            origin="BLR",
            destination="LHR",
            max_stops=1,
            baggage_required=True
        ),
        soft_preferences=SoftPreferences(
            optimization_goal="cheapest"
        )
    )

def test_filter_candidates(base_request):
    candidates = [
        Flight(airline="BA", flight_number="1", stops=0, baggage_included=True),
        Flight(airline="EK", flight_number="2", stops=2, baggage_included=True),  # Rejected: max stops
        Flight(airline="RY", flight_number="3", stops=0, baggage_included=False), # Rejected: no baggage
    ]
    
    valid = filter_candidates(candidates, base_request)
    assert len(valid) == 1
    assert valid[0].airline == "BA"

def test_deduplicate_candidates():
    candidates = [
        Flight(airline="BA", flight_number="1", price=1000, stops=0),
        Flight(airline="BA", flight_number="1", price=1000, stops=0), # Duplicate
        Flight(airline="BA", flight_number="1", price=1200, stops=0), # Different price, keep
        Flight(airline="EK", flight_number="2", price=1000, stops=0), # Different airline
    ]
    
    deduped = deduplicate_candidates(candidates)
    assert len(deduped) == 3

def test_rank_candidates_cheapest(base_request):
    base_request.soft_preferences.optimization_goal = "cheapest"
    candidates = [
        Flight(airline="BA", flight_number="1", price=1500, duration_minutes=600),
        Flight(airline="EK", flight_number="2", price=1000, duration_minutes=800),
    ]
    
    ranked = rank_candidates(candidates, base_request)
    assert ranked[0].airline == "EK" # Cheapest first

def test_rank_candidates_fastest(base_request):
    base_request.soft_preferences.optimization_goal = "fastest"
    candidates = [
        Flight(airline="BA", flight_number="1", price=1500, duration_minutes=600),
        Flight(airline="EK", flight_number="2", price=1000, duration_minutes=800),
    ]
    
    ranked = rank_candidates(candidates, base_request)
    assert ranked[0].airline == "BA" # Fastest first

def test_evidence_score():
    f_poor = Flight(airline="BA")
    f_good = Flight(airline="BA", flight_number="1", price=1000, stops=0, duration_minutes=600, source_url="example.com")
    
    assert f_poor.evidence_score == 0 # Only airline (needs flight_number too)
    assert f_good.evidence_score == 5
