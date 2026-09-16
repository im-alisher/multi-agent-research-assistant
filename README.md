# Multi-Agent Research Assistant

A multi-agent system built with LangGraph and LangChain where specialized agents collaborate to research and answer user questions.

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ Orchestrator │ ── breaks query into sub-tasks
└──────┬──────┘
       │
       ▼ (for each sub-task)
┌─────────────┐
│   Search     │ ── fetches raw web results
└──────┬──────┘
       ▼
┌─────────────┐
│  Summarizer  │ ── condenses results into notes
└──────┬──────┘
       ▼
┌─────────────┐
│ Fact-Checker │ ── flags unsupported claims
└──────┬──────┘
       │
       ▼ (once all sub-tasks processed)
┌─────────────┐
│   Writer     │ ── composes final answer
└─────────────┘
```

## Agents

- **Orchestrator**: Decomposes the query and controls routing through the pipeline
- **Search Agent**: Uses DuckDuckGo to find web results for each sub-task
- **Summarizer**: Uses an LLM to condense raw search results into concise notes
- **Fact-Checker**: Reviews summaries against raw results to flag unsupported claims
- **Writer**: Combines all verified notes into a coherent final answer

## Setup

```bash
# Clone and enter the project
cd multi-agent-research-assistant

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY (get one at https://console.groq.com)

# Run
python main.py
```

## Tests

```bash
pytest tests/ -v
```

## Tech Stack

- Python 3.11+
- LangGraph (state machine and routing)
- LangChain + Groq (LLM inference via llama-3.3-70b-versatile)
- DuckDuckGo Search (web search tool)
- python-dotenv (environment management)

## What I Learned

<!-- Fill in after completing the project -->

## What I Would Improve

<!-- Fill in after completing the project -->