import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List, Optional

class MockFlight(BaseModel):
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    price: Optional[float] = None
    stops: Optional[int] = None
    duration_minutes: Optional[int] = None

class FlightExtraction(BaseModel):
    flights: List[MockFlight]

MOCK_TAVILY_RESULTS = """
Search results for flights from BLR to LHR:
1. British Airways flight BA118, direct, costs $850. Duration is 11 hours and 15 mins.
2. Emirates flight EK533 via Dubai, 1 stop. Total price is $620. Duration 14 hours.
3. Lufthansa flight LH755, 1 stop in Frankfurt. Price $710.
4. Some random blog post about traveling to London.
"""

def run_extraction(enable_thinking: bool):
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    
    model_kwargs = {}
    if enable_thinking:
        model_kwargs = {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 512
            }
        }
    else:
        # Explicitly disable thinking
        model_kwargs = {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
    
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        max_tokens=1024,
        model_kwargs=model_kwargs
    )
    
    structured_llm = llm.with_structured_output(FlightExtraction)
    prompt = f"Extract the flights from these results:\n{MOCK_TAVILY_RESULTS}"
    
    print(f"\n--- Running with enable_thinking = {enable_thinking} ---")
    start_time = time.time()
    try:
        res = structured_llm.invoke([HumanMessage(content=prompt)])
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Extracted {len(res.flights)} flights:")
        for f in res.flights:
            print(f" - {f.airline}: ${f.price}, {f.stops} stops")
    except Exception as e:
        end_time = time.time()
        print(f"Failed after {end_time - start_time:.2f} seconds. Error: {e}")

if __name__ == "__main__":
    run_extraction(enable_thinking=True)
    run_extraction(enable_thinking=False)
