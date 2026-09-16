"""
agents/search_agent.py — The Search node.

This node is deliberately dumb by design: it reads the sub-task the
Orchestrator assigned (``state["current_subtask"]``), runs the web search
tool, and writes the raw results back into the state under that sub-task's key.

It performs no reasoning — no LLM is involved. All the judgement (which facts
matter, how to phrase the query) is delegated either to the Orchestrator
(choosing the sub-task) or to the Summarizer (distilling the results).

Because the search function is *injected* (passed in as a parameter) rather
than imported, unit tests can substitute a fake search function and never hit
the live web. Production code passes ``tools.search_tool.web_search``.
"""

from __future__ import annotations

from typing import Any, Callable

from graph.state import ResearchState, SearchResult
from tools.search_tool import SearchToolCallable, web_search  # re-exported for convenience


def make_search_node(
    search_func: SearchToolCallable = web_search,
) -> Callable[[ResearchState], dict[str, Any]]:
    """
    Return a LangGraph node that searches the web for the current sub-task.

    Args:
        search_func: A callable ``(query: str, max_results: int) -> list[SearchResult]``.
            Defaults to the real ``web_search`` wrapper.  Tests inject a stub.

    Returns:
        The node function ``(state: ResearchState) -> dict[str, Any]``, where the
        returned dict is the partial state update written back by LangGraph.
    """
    def search_node(state: ResearchState) -> dict[str, Any]:
        subtask = state.get("current_subtask", "")

        # Safety net: if the graph was invoked with no assigned sub-task (e.g.
        # a manual/incorrect entry point), do nothing rather than crash.
        if not subtask:
            print("[search_agent] no current_subtask in state; skipping search")
            return {"current_step": "search"}

        results: list[SearchResult] = search_func(subtask)

        # Key by the sub-task so the Summarizer / Fact-Checker can find the
        # data later without re-deriving it. Empty result lists are stored as
        # well (key present) — that tells downstream nodes "we looked and found
        # nothing", which is meaningful information.
        return {
            "current_step": "search",
            "raw_search_results": {subtask: results},
        }

    return search_node