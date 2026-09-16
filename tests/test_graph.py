"""
tests/test_graph.py — Integration tests for the assembled graph.

The whole compiled graph is exercised below with a mocked LLM and a mocked
search callable, so no real API traffic ever happens. These tests verify the
*routing*, not the output quality: subtasks get created, each one flows through
search → summarizer → fact-checker, they all get marked checked, and finally
the writer runs.
"""

from __future__ import annotations

from graph.build_graph import build_graph
from tests.fakes import FakeLLM, fake_search


def _initial_state(original_query: str = "Who created LangGraph?") -> dict:
    return {
        "original_query": original_query,
        "subtasks": [],
        "raw_search_results": {},
        "summarized_notes": {},
        "fact_check_flags": [],
        "checked_subtasks": [],
        "current_subtask": "",
        "final_answer": "",
        "current_step": "",
    }


class TestGraphEndToEnd:
    def test_full_pipeline_with_two_subtasks(self):
        graph = build_graph(FakeLLM(), fake_search)

        final = graph.invoke(_initial_state())

        # Orchestrator decomposed the query into two sub-tasks...
        assert len(final["subtasks"]) == 2

        # ...and both were fully processed (search + summarise + fact-check).
        assert sorted(final["checked_subtasks"]) == sorted(final["subtasks"])
        for subtask in final["subtasks"]:
            assert subtask in final["raw_search_results"]
            assert subtask in final["summarized_notes"]

        # The writer ran last and produced a final answer.
        assert final["current_step"] == "writer"
        assert final["final_answer"]

    def test_fact_check_flags_accumulate_across_subtasks(self):
        # One flag per sub-task: two flags expected in the final state.
        graph = build_graph(
            FakeLLM(flags=["unsupported claim"]),
            fake_search,
        )
        final = graph.invoke(_initial_state())
        assert len(final["fact_check_flags"]) == 2

    def test_routing_stops_when_no_subtasks_remain(self):
        # A decomposition that yields nothing should route straight to the
        # writer, which must still produce *some* answer (the "no notes" case).
        graph = build_graph(FakeLLM(subtasks=[]), fake_search)
        final = graph.invoke(_initial_state())
        assert final["current_step"] == "writer"
        assert final["subtasks"] == []
        assert final["final_answer"]

    def test_graph_is_reusable_across_separate_invocations(self):
        # Compiled graphs keep no per-run state, so two calls should be
        # independent (mutating state must not leak between them).
        graph = build_graph(FakeLLM(), fake_search)
        first = graph.invoke(_initial_state())
        second = graph.invoke(_initial_state("A totally different question"))
        assert second["original_query"] == "A totally different question"
        assert first["subtasks"] == second["subtasks"]

    def test_no_real_search_or_llm_called(self, monkeypatch):
        """Guarding the test suite's key promise: no live API traffic."""
        def fail(*a, **k):
            raise AssertionError("LIVE SEARCH WAS CALLED — tests must stay offline")

        with monkeypatch.context() as ctx:
            # If the graph somehow fell back to the real tool, this explodes.
            ctx.setattr("tools.search_tool.DDGS", fail)
            graph = build_graph(FakeLLM(), fake_search)
            final = graph.invoke(_initial_state())
            assert final["final_answer"]