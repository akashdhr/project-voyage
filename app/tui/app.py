from textual.app import App, ComposeResult
from textual.containers import Container, ScrollableContainer
from textual.widgets import Header, Footer, Input, RichLog, Static
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from app.agent.graph import build_graph

# Load environment variables from .env file
load_dotenv()

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
    #log-container {
        height: 1fr;
    }
    """

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="input-container"):
            yield Input(placeholder="e.g., Cheapest flight from BLR to London in October, max 1 stop", id="query-input")
        with ScrollableContainer(id="log-container"):
            yield RichLog(id="agent-log", highlight=True, markup=True)
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value
        log = self.query_one(RichLog)
        
        if not query.strip():
            return
            
        log.write(f"\n[bold green]User:[/bold green] {query}")
        self.query_one(Input).value = ""
        
        # Run agent
        log.write("[bold blue]🚀 Starting Voyage Agent...[/bold blue]")
        
        graph = build_graph()
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "iteration_count": 0,
            "candidate_flights": []
        }
        
        try:
            async for event in graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                name = event["name"]
                
                if kind == "on_chain_start" and name in ["parse_request", "search", "extract_candidates", "evaluate", "rank", "recommend"]:
                    log.write(f"[cyan]▶ Node Started:[/cyan] {name}")
                    
                elif kind == "on_chain_end" and name in ["parse_request", "search", "extract_candidates", "evaluate", "rank", "recommend"]:
                    data = event.get("data", {}).get("output", {})
                    
                    if isinstance(data, dict):
                        if "search_feedback" in data and data["search_feedback"]:
                            log.write(f"  [bold yellow]⚠️ Search Expansion Triggered:[/bold yellow] {data['search_feedback']}")
                        
                        if "candidate_flights" in data:
                            num = len(data["candidate_flights"])
                            log.write(f"  [dim]Tracked {num} valid flights so far[/dim]")
                            
                        if "recommendation" in data:
                            log.write(f"\n[bold magenta]Final Recommendation:[/bold magenta]\n{data['recommendation']}")
                            
                    log.write(f"[green]✔ Node Completed:[/green] {name}")
                    
                elif kind == "on_tool_start":
                    tool_input = event.get("data", {}).get("input")
                    log.write(f"  [bold blue]🛠️ Calling Tool:[/bold blue] {name} with args {tool_input}")
                    
        except Exception as e:
            log.write(f"[bold red]Error:[/bold red] {str(e)}")

def main():
    app = VoyageTUI()
    app.run()

if __name__ == "__main__":
    main()
