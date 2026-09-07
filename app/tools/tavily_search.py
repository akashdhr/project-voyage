import os
from typing import Optional
from langchain_core.tools import tool
from tavily import TavilyClient

@tool
def search_flights_tavily(query: str) -> str:
    """
    Search the web for flights. Use this to find flight prices, schedules, and options.
    Provide a specific query like 'cheapest flights from Bangalore to London next week one stop'.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not set."
    
    try:
        client = TavilyClient(api_key=api_key)
        # We use standard search with high depth to get flight info
        response = client.search(
            query=query, 
            search_depth="advanced",
            max_results=2
        )
        
        # Format the results into a readable string for the LLM
        results = []
        for result in response.get("results", []):
            results.append(f"Source: {result.get('url')}\nContent: {result.get('content')}\n")
            
        if not results:
            return "No relevant flight information found."
            
        return "\n---\n".join(results)
    except Exception as e:
        return f"Error executing Tavily search: {str(e)}"
