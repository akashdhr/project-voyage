# Project Voyage

Agentic Flight Research & Price Optimization Engine

## Problem

Building LLM applications involves more than just sending text to an API. It requires structured reasoning, tool calling, state management, and strict separation between semantic tasks (LLMs) and deterministic tasks (Python). This project serves as a practical, fundamentals-first exploration of Agentic AI.

## Architecture & Agent Workflow

The agent operates in a bounded loop governed by LangGraph:
1. **parse_request**: Understands natural language constraints and outputs JSON.
2. **search**: Generates search queries and calls Tavily.
3. **extract_candidates**: Parses messy markdown results into a strict `Flight` model.
4. **evaluate**: Decides if sufficient information exists or if search needs expansion.
5. **rank**: Deterministically filters out flights violating constraints (e.g. stops) and ranks by price.
6. **recommend**: Explains the final choice.

## Frameworks & Tradeoffs

### LangChain
**What problem does it solve?** Standardization of LLM interfaces, structured output parsing, and tool definition.
**Without it?** We would write raw `requests` calls to OpenAI-compatible endpoints, manually handle JSON schema generation for tool definitions, and manually parse stringified JSON tool calls.
**Why introduced?** To drastically reduce boilerplate when defining the Tavily tool and structured data extractors.

### LangGraph
**What problem does it solve?** Agent orchestration, cyclic state management, and routing.
**Without it?** We would use a massive Python `while` loop with nested `if/else` statements that is hard to debug, trace, or pause/resume.
**Why introduced?** Phase 1's `while` loop becomes unmaintainable as soon as we introduce complex evaluation routing (e.g., branching to "rank" vs "search_again"). LangGraph formalizes this as a state machine.

### LangSmith (Future Phase)
**What problem does it solve?** Observability and Evaluation.
**Without it?** We rely on stdout console logs, which become impossible to read when tracking multiple iterations of large context windows.
**Why introduced?** To see the exact trace of "LLM Prompt -> Tool Call -> Tool Result" and to run quantitative evaluations.

### Why Tavily? Why not SerpAPI initially?
Tavily is an LLM-optimized search engine that returns clean text/markdown instead of raw HTML. We use it exclusively because the primary goal of this project is to learn *Agentic AI* and *web information retrieval*. If we started with SerpAPI (a structured flights API), we would bypass the core challenge of having the LLM reason about messy, unstructured data. We will only introduce a structured provider like SerpAPI later if we prove that Tavily has a genuine limitation (e.g., inability to find reliable real-time prices) that blocks the agent from succeeding.

## Setup & Testing

1. Create virtual environment: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -e .[dev]`
3. Config: Copy `.env.example` to `.env` and add API keys.
4. Run TUI: `python -m app.tui`
5. Test: `pytest tests/unit/` (Unit tests run deterministically without network).

*Note: Discovered prices are unverified discovery information.*