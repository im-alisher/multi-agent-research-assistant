"""
graph/state.py — The shared state schema for the Multi-Agent Research Assistant graph.

In LangGraph, agents do NOT call each other directly. Instead every agent is a
"node" in a graph, and all of the information agents produce is written into a
single shared *state* object that flows through the graph. Nodes read whatever
they need from the state and write their results back into it. The graph
framework handles passing the updated state from one node to the next.

This module defines:

1. ``SearchResult`` — a single hit returned by the web search tool.
2. ``ResearchState`` — the TypedDict schema describing the entire shared state.
3. A couple of type aliases used across the project.

The reducer functions (``Annotated`` + ``operator.add``) are worth a closer look.
By default, when a node returns a value for a key, LangGraph *replaces* whatever
was stored before. That is fine for fields like ``final_answer`` that we only
set once, but it is wrong for list-like fields such as ``subtasks`` or
``fact_check_flags`` that are appended to over multiple node executions.

Annotated[T, reducer] tells LangGraph: "instead of replacing, apply this reducer".
For lists we use ``operator.add``, which concatenates the new list onto the
existing one. This is exactly how the agents accumulate results as the graph
loops over sub-tasks.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class SearchResult(TypedDict):
    """
    A single search result returned by ``tools/search_tool.py``.

    Keeping this as a TypedDict (rather than an ad-hoc dict) gives us a stable
    contract between the Search agent and the web search tool, and lets editors /
    type checkers catch mismatches early.
    """

    title: str
    snippet: str
    url: str


class ResearchState(TypedDict):
    """
    The shared state that flows through the entire research graph.

    Every field is deliberately optional with a sensible default of ``None`` /
    empty so that any node can be tested in isolation and so the graph can be
    invoked incrementally (e.g. in tests) without pre-filling everything.

    Field routing summary:
      * ``original_query``   — set once by the user, read by every agent.
      * ``subtasks``         — set (accumulated) by the Orchestrator.
      * ``raw_search_results`` — keyed by sub-task, written by the Search agent.
      * ``summarized_notes`` — keyed by sub-task, written by the Summarizer.
      * ``fact_check_flags`` — appended by the Fact-Checker.
      * ``final_answer``     — written once, by the Writer.
      * ``current_step``     — ""returns"" the name of the node that last ran;
                               mainly for logging/debugging the flow.
    """

    # The question the user asked. Singular, immutable once set.
    original_query: str

    # The concrete sub-questions the Orchestrator decomposed the query into.
    # Uses a reducer so sub-tasks accumulate rather than clobber each other.
    subtasks: Annotated[list[str], operator.add]

    # Raw web results keyed by the sub-task they answer.
    # e.g. {"What is X?": [{"title": ..., "snippet": ..., "url": ...}, ...]}
    raw_search_results: dict[str, list[SearchResult]]

    # Concise LLM summaries keyed by the same sub-task keys as above.
    summarized_notes: dict[str, str]

    # Flags raised by the Fact-Checker when a summary seems unsupported,
    # exaggerated, or contradictory. Appended with operator.add.
    fact_check_flags: Annotated[list[str], operator.add]

    # The final assembled answer produced by the Writer.
    final_answer: str

    # Human-readable marker of where the graph currently is in the flow.
    # Useful for the communication logger and for debugging a run.
    current_step: str