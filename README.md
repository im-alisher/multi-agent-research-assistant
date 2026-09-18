# Multi-Agent Research Assistant

A multi-agent system built with **LangGraph** and **LangChain** where a user
submits a research question and five specialized agents collaborate to answer
it. The agents do **not** call each other directly — they communicate by
reading and writing a shared *state* object as LangGraph routes execution
between them.

This is a learning / portfolio project, so the code favors clear structure,
explicit dependency injection, and heavy commenting over cleverness or
premature optimization.

## How it works (architecture)

```
                       ┌────────────────────┐
User question ───────▶ │    ORCHESTRATOR     │  decomposes query into 2–4
                       └─────────┬──────────┘  sub-questions; hands them out
                                 │
           (conditional edge: dispatch)
                                 │
                   ┌─────────────┴──────────┐
                   ▼                        ▼          ...once all sub-tasks are
                SEARCH                 (all done)     done, the same edge
                   │                     │            routes to...
                   ▼                     ▼
             SUMMARIZER                WRITER ───▶ Final answer
                   │
                   ▼
           FACT-CHECKER ──── back to ORCHESTRATOR for next sub-task
```

Each sub-task flows through the same pipeline:

1. **Orchestrator** — receives the user's question, asks the LLM to break it
   into 2–4 concrete, searchable sub-questions, and stores them in state. Every
   time it is re-entered it "hands out" the next unfinished sub-task by writing
   `current_subtask` into the state (a *dispatcher* pattern).
2. **Search** — no LLM here; calls the DuckDuckGo web-search wrapper for the
   current sub-task and stores raw results keyed by that sub-task.
3. **Summarizer** — condenses the raw snippets into a short factual note with
   its sources, keyed by the same sub-task.
4. **Fact-Checker** — re-reads the summary *against the raw results* and flags
   anything unsupported, exaggerated, or contradictory. Flags accumulate in
   state, and the sub-task is marked "checked".
5. **Writer** — once every sub-task is checked, composes the final structured
   answer from all notes, caveating any flagged claims, and adds a Sources
   section.

### How the agents actually "talk" to each other

The key idea is that a LangGraph node is just a function
`(state) -> partial_state_update`. None of the five agents import each other or
hold references to one another — they only know about the `ResearchState`
schema (`graph/state.py`). When a node returns a dict like
`{"summarized_notes": {"Why X?": "..."}}`, LangGraph *merges* that update back
into the shared state and then routes to the next node according to the edges
you declared. The next node reads whatever fields it needs.

Because list/dict fields get *reducers* (see below), multiple agents can append
to the same field across a loop without clobbering each other — this is how
`fact_check_flags`, `subtasks` and the per-sub-task result dicts accumulate.

The routing is controlled by a **conditional edge** after the Orchestrator and
again after the Fact-Checker. Both call the same pure function `dispatch`
(`agents/orchestrator.py`), which reads `current_subtask`:

* `current_subtask` set → route to **search** (keep processing sub-tasks);
* `current_subtask` empty (no pending sub-tasks left) → route to **writer**.

### State schema and reducers

`graph/state.py` defines the shared `ResearchState` TypedDict:

| Field | Written by | Notes |
| --- | --- | --- |
| `original_query` | user | set once at start |
| `subtasks` | orchestrator | `operator.add` — accumulates |
| `current_subtask` | orchestrator | overwritten each dispatch |
| `raw_search_results` | search | `operator.or_` — dict-merged per sub-task |
| `summarized_notes` | summarizer | `operator.or_` — dict-merged per sub-task |
| `fact_check_flags` | fact-checker | `operator.add` — accumulates |
| `checked_subtasks` | fact-checker | `operator.add` — drives completion |
| `final_answer` | writer | set once at the end |
| `current_step` | every agent | human-readable "where are we" tracker (for logs) |

The `Annotated[T, reducer]` annotations tell LangGraph *how* to fold a node's
update into the field. Without a reducer, a node returning a dict field would
**replace** the whole field — which would silently drop earlier sub-tasks'
results. `operator.add` concatenates lists; `operator.or_` merges dicts.

### Communication logging

Every node in `build_graph` is wrapped in `graph/logger.py`, so each transition
writes what agent ran, what state it received, and what it produced to
`logs/agent_comm.log`. Running `main.py` also snapshots a per-run copy to
`logs/run_<timestamp>.log`. Read a log file to literally watch the state pass
from agent to agent.

## Agent evaluation framework

The pipeline *runs* agents, but until now nothing measured *how well they ran*.
The `evaluation/` package fixes that: after a run completes, `main.py` inspects
the final state and the run's communication log and emits a per-agent
`pass` / `warning` report. It is deliberately tiny and transparent — a
teaching-grade "how did each agent behave?" harness, not a research benchmark.

```
final state + this run's log entries
    ──collect──▶ RunMetrics        (evaluation/collector.py — pure measurements)
    ──score────▶ Verdicts          (evaluation/scorer.py   — if/then rules)
    ──report──▶ markdown           (evaluation/report.py   — console + reports/)
```

Skip the pass for a faster run with `python main.py --no-eval "..."`.

### What it measures

| Agent | Measured | Rule (see `scorer.py`) |
| --- | --- | --- |
| Orchestrator | sub-task count, loop termination | warn if count is outside 2–4, or not every sub-task was marked "checked" (a stuck/infinite loop symptom) |
| Search | number of calls, per-sub-task result counts | warn if any sub-task got zero results |
| Summarizer | summaries that condense, non-triviality | warn if a note is as long as its raw input, empty, or trivial (< 20 chars) |
| Fact-checker | total flags + per-sub-task attribution | informational warn if flags were raised on more than half of the sub-tasks |
| Writer | sub-task coverage, answer length | warn if the answer never mentions a sub-task (exact text or its significant words), or is shorter than 100 / longer than 20 000 chars |
| Overall | run time, node transitions, success | warn if the graph errored out or nothing ran; runtime/`transitions` always reported |

### Why rule-based scoring?

The same numbers could drive a machine-learned regressor or an "LLM judge"
rubric, but this project chose hand-written rules on purpose — for the same
reason the rest of the codebase favors clarity over cleverness:

1. **Explainable.** Every verdict is a readable if/then ("writer warning:
   answer is 12 chars, minimum is 100"). There is no black box to trust on
   faith — you can audit the whole judgement policy in one file.
2. **Cheap and deterministic.** No extra API calls, no randomness; the same run
   always yields the same report.
3. **Measurement is decoupled from judgement.** `collector.py` stores only
   facts; `scorer.py` holds all opinion. Making the rules smarter later never
   requires replaying runs or touching agents.
4. **Honest about uncertainty.** Measurements here are heuristics (e.g. "is
   the sub-task mentioned in the answer?" is approximate for a paraphrasing
   LLM), so the verdicts stay coarse — `pass`/`warning`, never a fake-out
   "88.3 out of 100" score.

**Known limits:** the framework evaluates *agent behavior* (completeness,
condensation, coverage, flow), not the *semantic quality* of the final answer
(whether the facts are correct or well argued). Flagging upstream
summaries is probably the most useful signal here. Attribution of fact-check
flags to sub-tasks is inferred from visit order — a subtle assumption, and one
of the first things a more rigorous design should change (e.g. by tagging the
sub-task inside each flag).

### Example report

```
# Agent Evaluation Report

- **Query:** Who created LangGraph?
- **Run at:** 2026-09-18 15:20:43
- **Completed:** yes
- **Total run time:** 6.20s
- **Node transitions:** 11

## Verdicts

| Agent | Status | Explanation |
| --- | --- | --- |
| orchestrator | PASS | produced 2 sub-task(s) and the dispatcher loop terminated cleanly. |
| search       | PASS | made 2 search call(s); every sub-task yielded results. |
| summarizer   | PASS | condensed every sub-task's raw results into a non-trivial note. |
| fact_checker | PASS | raised 0 flag(s); within expected ratio. |
| writer       | PASS | answer covers every sub-task and is 1024 chars long. |
| overall      | PASS | graph completed successfully in 6.20s across 11 node transition(s). |

## Measurements

| Sub-task | Results | Flagged | In answer |
| --- | --- | --- | --- |
| What is LangGraph?       | 5 | 0 | yes |
| Who created LangGraph?  | 5 | 0 | yes |
```

A copy lands in `reports/eval_<timestamp>.md` for every evaluated run.

## Project layout

```
multi-agent-research-assistant/
  agents/
    llm.py              shared Groq LLM factory
    orchestrator.py     decomposition + dispatch routing
    search_agent.py     runs the web search for the current sub-task
    summarizer_agent.py condenses raw results into a note
    fact_checker_agent.py flags unsupported/exaggerated claims
    writer_agent.py     composes the final answer
  graph/
    state.py            shared state schema (TypedDict + reducers)
    build_graph.py      wires all nodes and edges, compiles the graph
    logger.py           inter-agent communication logger
  tools/
    search_tool.py      DuckDuckGo wrapper (mockable, in tests)
  evaluation/
    metrics.py          RunMetrics data model (measurements, not judgements)
    collector.py        derives RunMetrics from final state + log entries
    scorer.py           hand-written pass/warning rules per agent
    report.py           renders the report to console + reports/ as markdown
  tests/                unit + end-to-end tests (offline, mocked)
  logs/                 communication logs (gitignored)
  reports/              per-run evaluation reports (gitignored)
  main.py               CLI entry point (--no-eval skips evaluation)
  requirements.txt
  .env.example
```

## Setup

Requirements: **Python 3.11+** (developed against 3.13).

```bash
# 1. Virtual environment and dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. API key (only needed for real runs, not for tests)
cp .env.example .env
#    edit .env and set GROQ_API_KEY=your-key   (https://console.groq.com)

# 3. Run
python main.py "Why did NASA select Dragonfly for Titan?"
#    or without an argument for an interactive prompt
```

## Tests

The whole test suite runs **offline** — the LLM and search tool are mocked:

```bash
pytest tests/ -v
```

`tests/test_graph.py` includes a full end-to-end run of the compiled
graph with a fake LLM and fake search, plus a guard test asserting no live
search is ever triggered during testing.

## What I Learned

<!-- Fill this in yourself: e.g. how LangGraph merges state updates, why
reducers exist, how a dispatcher loop works, DI for testability, etc. -->

## What I Would Improve

<!-- Fill this in yourself: e.g. parallelizing searches per sub-task,
retry/backoff on API errors, persisting state, adding Tavily, semantic
evaluation of the final ANSWER (the evaluation/ framework covers agent
BEHAVIOR, not answer quality), a supervisor agent with human-in-the-loop, etc. -->

## Tech stack

- Python 3.11+
- LangGraph — graph orchestration, state chaining, conditional routing
- LangChain — prompts, chat-model wrappers, structured output
- LangChain-Groq — `llama-3.3-70b-versatile` via the Groq free tier
- `ddgs` (DuckDuckGo) — web search, wrapped in `tools/search_tool.py`
- python-dotenv — environment/API-key management
- pytest — offline unit and integration tests