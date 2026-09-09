"""Run a live end-to-end smoke test against the configured external services."""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app.agent.graph import build_graph


async def run():
    load_dotenv()
    graph = build_graph()

    query = "cheapest flight from NYC to LON for tomorrow. 1 stop max."
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "iteration_count": 0,
        "candidate_flights": [],
    }

    print("Starting live smoke test...")
    try:
        final_state = await graph.ainvoke(initial_state)
        print("Success! Recommended flight:")
        print(final_state.get("recommendation", "None found."))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run())
