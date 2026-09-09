import json
import asyncio
from typing import Optional, List, Dict, Any
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Input, RichLog, Static, DataTable
from textual.reactive import reactive
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from app.agent.graph import build_graph
from app.domain.models import Flight, FlightSearchRequest

load_dotenv()

class DemoStreamer:
    """Simulates an event stream for demo mode."""
    def __init__(self, query: str):
        self.query = query

    async def astream_events(self, initial_state, version="v2"):
        events = [
            {"event": "on_chain_start", "name": "parse_request"},
            {"event": "on_chain_end", "name": "parse_request", "data": {"output": {"request": {"origin": "BLR", "destination": "LHR"}}}},
            {"event": "on_chain_start", "name": "search"},
            {"event": "on_tool_start", "name": "search_flights_tavily", "data": {"input": {"query": "BLR London October flights checked baggage"}}},
            {"event": "on_tool_end", "name": "search_flights_tavily", "data": {"output": [{"url": "example.com"}]}},
            {"event": "on_chain_end", "name": "search"},
            {"event": "on_chain_start", "name": "extract_candidates"},
            {"event": "on_chain_end", "name": "extract_candidates", "data": {"output": {"candidate_flights": [
                Flight(airline="BA", flight_number="118", origin_airport="BLR", destination_airport="LHR", price=72400.0, currency="INR", stops=1, duration_minutes=700, baggage_included=True)
            ]}}},
            {"event": "on_chain_start", "name": "evaluate"},
            {"event": "on_chain_end", "name": "evaluate", "data": {"output": {"search_feedback": "Only 1 candidate found. Expanding search to LGW."}}},
            {"event": "on_chain_start", "name": "search"},
            {"event": "on_tool_start", "name": "search_flights_tavily", "data": {"input": {"query": "BLR LGW October flights checked baggage"}}},
            {"event": "on_tool_end", "name": "search_flights_tavily", "data": {"output": [{"url": "example.com"}]}},
            {"event": "on_chain_end", "name": "search"},
            {"event": "on_chain_start", "name": "extract_candidates"},
            {"event": "on_chain_end", "name": "extract_candidates", "data": {"output": {"candidate_flights": [
                Flight(airline="BA", flight_number="118", origin_airport="BLR", destination_airport="LHR", price=72400.0, currency="INR", stops=1, duration_minutes=700, baggage_included=True),
                Flight(airline="EK", flight_number="502", origin_airport="BLR", destination_airport="LGW", price=75200.0, currency="INR", stops=1, duration_minutes=730, baggage_included=True),
                Flight(airline="QR", flight_number="573", origin_airport="BLR", destination_airport="LHR", price=69800.0, currency="INR", stops=2, duration_minutes=655, baggage_included=True)
            ]}}},
            {"event": "on_chain_start", "name": "evaluate"},
            {"event": "on_chain_end", "name": "evaluate", "data": {"output": {"search_feedback": None}}},
            {"event": "on_chain_start", "name": "rank"},
            {"event": "on_chain_end", "name": "rank", "data": {"output": {"candidate_flights": [
                Flight(airline="QR", flight_number="573", origin_airport="BLR", destination_airport="LHR", price=69800.0, currency="INR", stops=2, duration_minutes=655, baggage_included=True),
                Flight(airline="BA", flight_number="118", origin_airport="BLR", destination_airport="LHR", price=72400.0, currency="INR", stops=1, duration_minutes=700, baggage_included=True),
                Flight(airline="EK", flight_number="502", origin_airport="BLR", destination_airport="LGW", price=75200.0, currency="INR", stops=1, duration_minutes=730, baggage_included=True)
            ]}}},
            {"event": "on_chain_start", "name": "recommend"},
            {"event": "on_chain_end", "name": "recommend", "data": {"output": {"recommendation": "I recommend QR 573 as the cheapest option, but BA 118 is faster with fewer stops."}}},
        ]
        for e in events:
            yield e
            await asyncio.sleep(0.5)

class ReplayStreamer:
    """Reads events from a JSON file and streams them."""
    def __init__(self, filepath: str):
        self.filepath = filepath

    async def astream_events(self, initial_state, version="v2"):
        with open(self.filepath, 'r') as f:
            events = json.load(f)
        for e in events:
            yield e
            await asyncio.sleep(0.1)

class VoyageTUI(App):
    """A Textual TUI for Project Voyage."""

    CSS = """
    Screen {
        layout: vertical;
    }
    #input-container {
        height: 3;
        dock: top;
    }
    #main-container {
        layout: horizontal;
        height: 1fr;
    }
    #left-pane {
        width: 35%;
        height: 100%;
        border: solid green;
    }
    #right-pane {
        width: 65%;
        height: 100%;
        border: solid blue;
    }
    #candidates-container {
        height: 12;
        dock: bottom;
        border: solid yellow;
    }
    """

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit", "Quit")]

    def __init__(self, demo_mode: bool = False, replay_file: Optional[str] = None):
        super().__init__()
        self.demo_mode = demo_mode
        self.replay_file = replay_file
        self.iteration_count = 0
        self.max_iterations = 3
        self.stages = {
            "parse_request": "Parse request",
            "search": "Generate search strategy",
            "search_flights_tavily": "Search web",
            "extract_candidates": "Extract candidates",
            "evaluate": "Evaluate results",
            "rank": "Rank flights",
            "recommend": "Generate recommendation"
        }

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="input-container"):
            yield Input(placeholder="e.g., Cheapest flight from BLR to London in October...", id="query-input")
        
        with Container(id="main-container"):
            with Vertical(id="left-pane"):
                yield Static(" [bold]Agent Status[/bold]", id="status-header")
                yield RichLog(id="status-log", highlight=True, markup=True, wrap=True)
                yield Static("\n [bold]Search Iterations[/bold]", id="iterations-header")
                yield RichLog(id="iterations-log", highlight=True, markup=True, wrap=True)
            
            with Vertical(id="right-pane"):
                yield Static(" [bold]Tool Activity & Reasoning[/bold]", id="tool-header")
                yield RichLog(id="tool-log", highlight=True, markup=True, wrap=True)

        with Container(id="candidates-container"):
            yield Static(" [bold]Candidate Summary[/bold]", id="candidates-header")
            yield DataTable(id="candidates-table")
        
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Flight", "Price", "Stops", "Duration", "Baggage")
        
        if self.replay_file:
            self.query_one(Input).value = f"[Replaying {self.replay_file}] Press Enter..."
        elif self.demo_mode:
            self.query_one(Input).value = "Find the cheapest flight from Bangalore to London..."

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value
        tool_log = self.query_one("#tool-log", RichLog)
        status_log = self.query_one("#status-log", RichLog)
        iter_log = self.query_one("#iterations-log", RichLog)
        table = self.query_one(DataTable)
        
        if not query.strip():
            return
            
        tool_log.clear()
        status_log.clear()
        iter_log.clear()
        table.clear()
        
        tool_log.write(f"[bold green]User Request:[/bold green]\n{query}\n")
        
        if not self.replay_file:
            self.query_one(Input).value = ""
            self.query_one(Input).disabled = True
        
        if self.demo_mode:
            graph = DemoStreamer(query)
        elif self.replay_file:
            graph = ReplayStreamer(self.replay_file)
        else:
            graph = build_graph()
            
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "iteration_count": 0,
            "candidate_flights": []
        }
        
        self.iteration_count = 1
        iter_log.write(f"Search iteration: {self.iteration_count} / {self.max_iterations}")
        
        try:
            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event["name"]
                data = event.get("data", {}).get("output", {})
                
                if kind == "on_chain_start" and name in self.stages:
                    status_log.write(f"[dim]⧖ {self.stages[name]}[/dim]")
                    
                elif kind == "on_chain_end" and name in self.stages:
                    status_log.write(f"[green]✓ {self.stages[name]}[/green]")
                    
                    if isinstance(data, dict):
                        if name == "evaluate" and "search_rationale" in data and data["search_rationale"]:
                            rationale = data["search_rationale"]
                            iter_log.write(f"[yellow]{rationale}[/yellow]")
                            if "search_feedback" in data and data["search_feedback"]:
                                status_log.write(f"[yellow]⚠ Insufficient valid candidates[/yellow]")
                                status_log.write(f"[cyan]→ Expanding search strategy[/cyan]")
                                self.iteration_count += 1
                                iter_log.write(f"\nSearch iteration: {self.iteration_count} / {self.max_iterations}")
                        
                        if "candidate_flights" in data and name in ["extract_candidates", "rank"]:
                            flights = data["candidate_flights"]
                            table.clear()
                            for f in flights:
                                # Convert dict to object if necessary (happens in replay mode JSON parsing)
                                if isinstance(f, dict):
                                    f = Flight(**f)
                                
                                flight_name = f"{f.airline} {f.flight_number}"
                                price = f"{f.currency} {f.price}" if f.price else "Unknown"
                                stops = str(f.stops)
                                duration = f"{f.duration_minutes // 60}h {f.duration_minutes % 60}m" if f.duration_minutes else "Unknown"
                                baggage = "✓" if f.baggage_included else "✗"
                                table.add_row(flight_name, price, stops, duration, baggage)
                            
                        if "recommendation" in data:
                            tool_log.write(f"\n[bold magenta]Final Recommendation:[/bold magenta]\n{data['recommendation']}")
                            
                elif kind == "on_tool_start":
                    status_log.write(f"[dim]⧖ {self.stages.get(name, name)}[/dim]")
                    tool_input = event.get("data", {}).get("input", {})
                    query = tool_input.get("query", str(tool_input))
                    tool_log.write(f"\n[bold blue]Tool:[/bold blue] {name.replace('_', ' ').title()}")
                    tool_log.write(f"[bold blue]Query:[/bold blue] {query}")
                    
                elif kind == "on_tool_end":
                    status_log.write(f"[green]✓ {self.stages.get(name, name)}[/green]")
                    tool_output = event.get("data", {}).get("output", "")
                    if isinstance(tool_output, list):
                        tool_log.write(f"[dim]Results:[/dim] {len(tool_output)}")
                    elif isinstance(tool_output, str):
                        tool_log.write(f"[dim]Search completed[/dim]")
                        
        except Exception as e:
            tool_log.write(f"\n[bold red]Error:[/bold red] {str(e)}")
        finally:
            self.query_one(Input).disabled = False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Project Voyage TUI")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without API calls")
    parser.add_argument("--replay", type=str, help="Path to a recorded run JSON file to replay")
    args = parser.parse_args()
    
    app = VoyageTUI(demo_mode=args.demo, replay_file=args.replay)
    app.run()

if __name__ == "__main__":
    main()
