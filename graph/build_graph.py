"""
graph/build_graph.py — Assembles the full multi-agent research graph.

This module wires the five agent nodes together with LangGraph's ``StateGraph``.
It is the only place in the codebase that knows about *all* agents — the agents
themselves do not import each other.

Graph topology (the "dispatcher" loop):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                        ┌────────────────────────────────────────┐        │
    │                        ▼                                        │        │
    │  ─────── ORCHESTRATOR ────(dispatch: SEARCH?)─── SEARCH ────────│        │
    │     │        ▲                           ▲     SUMMARIZER       │        │
    │     │        │                           ▼     FACT-CHECKER     │        │
    │     └────────┘                           │                       │        │
    │  (dispatch: WRITER?) ─── WRITER ─── END                          │        │
    └──────────────────────────────────────────────────────────────────────────┘

    • The Orchestrator is entered once to decompose the query, then re-entered
      once per sub-task to hand out the next pending one.
    • The conditional edge *after* the Orchestrator node is the ``dispatch``
      function: it returns ``"search"`` if ``current_subtask`` is non-empty,
      or ``"writer"`` if all sub-tasks have been checked off.
    • The same ``dispatch`` function is also wired after the Fact-Checker so
      that, once a sub-task finishes the search → summarise → fact-check
      pipeline, the graph loops back through Orchestrator → dispatch → next
      sub-task or Writer.

``build_graph(llm, search_fn)`` is the single public entry-point.  Passing
both ``llm`` and ``search_fn`` makes it trivial to inject fakes in tests.

``build_default_graph()`` is a convenience wrapper that uses the real LLM
from ``agents.llm`` and the real DuckDuckGo search tool.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from graph.state import ResearchState
from tools.search_tool import SearchToolCallable, web_search
from agents.llm import get_llm
from agents.orchestrator import (
    NODE_FACT_CHECKER,
    NODE_SEARCH,
    NODE_SUMMARIZER,
    NODE_WRITER,
    dispatch,
    make_orchestrator_node,
)
from agents.search_agent import make_search_node
from agents.summarizer_agent import make_summarizer_node
from agents.fact_checker_agent import make_fact_checker_node
from agents.writer_agent import make_writer_node
from graph.logger import wrap_for_logging


def build_graph(
    llm: BaseChatModel,
    search_fn: SearchToolCallable = web_search,
) -> StateGraph:
    """
    Build and return a **compiled** ``StateGraph`` ready to ``.invoke()``.

    Args:
        llm: Chat model instance used by Orchestrator, Summarizer, Fact-Checker,
             and Writer.  Inject a fake in tests.
        search_fn: The web search callable used by the Search node.
                   Accepts ``(query, max_results) -> list[SearchResult]``.

    Returns:
        The compiled ``StateGraph`` — call ``graph.invoke({"original_query": ...})``
        with a fresh state dict containing at minimum ``original_query``.

    Usage::

        from graph.build_graph import build_graph
        graph = build_graph(get_llm())
        result = graph.invoke({
            "original_query": "Compare LangGraph and CrewAI",
            "subtasks": [],
            "raw_search_results": {},
            "summarized_notes": {},
            "fact_check_flags": [],
            "checked_subtasks": [],
            "final_answer": "",
            "current_step": "",
            "current_subtask": "",
        })
    """
    # ── Build the nodes ──────────────────────────────────────────────────────
    orchestrator = make_orchestrator_node(llm)
    search       = make_search_node(search_fn)
    summarizer   = make_summarizer_node(llm)
    fact_checker = make_fact_checker_node(llm)
    writer       = make_writer_node(llm)

    # ── Assemble the graph ───────────────────────────────────────────────────
    graph = StateGraph(ResearchState)

    # Every node is wrapped in the communication logger so each state
    # transition is recorded to logs/agent_comm.log (see graph/logger.py).
    graph.add_node("orchestrator", wrap_for_logging("orchestrator", orchestrator))
    graph.add_node("search",       wrap_for_logging("search", search))
    graph.add_node("summarizer",   wrap_for_logging("summarizer", summarizer))
    graph.add_node("fact_checker", wrap_for_logging("fact_checker", fact_checker))
    graph.add_node("writer",       wrap_for_logging("writer", writer))

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("orchestrator")

    # ── Conditional edge: Orchestrator → Search or Writer ─────────────────────
    graph.add_conditional_edges("orchestrator", dispatch, {
        NODE_SEARCH: NODE_SEARCH,
        NODE_WRITER: NODE_WRITER,
    })

    # ── Per-subtask pipeline ─────────────────────────────────────────────────
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "fact_checker")

    # ── After Fact-Checker: back to Orchestrator to hand out next sub-task ───
    graph.add_conditional_edges("fact_checker", dispatch, {
        NODE_SEARCH: "orchestrator",
        NODE_WRITER: NODE_WRITER,
    })

    # ── Writer → END ─────────────────────────────────────────────────────────
    graph.add_edge("writer", END)

    return graph.compile()


def build_default_graph() -> Any:
    """
    Convenience factory that uses the real Groq LLM and DuckDuckGo search.

    Returns:
        The compiled graph, ready to ``.invoke()``.
    """
    return build_graph(llm=get_llm(), search_fn=web_search)