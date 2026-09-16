"""
tests/test_agents.py — Unit tests for the state schema, search tool, and each
individual agent node, in isolation (no real LLM or web access).
"""

from __future__ import annotations

import pytest

from graph.state import ResearchState, SearchResult
from tools import search_tool
from agents.orchestrator import (
    NODE_SEARCH,
    NODE_WRITER,
    SubtaskPlan,
    dispatch,
    make_orchestrator_node,
)
from agents.search_agent import make_search_node
from agents.summarizer_agent import make_summarizer_node
from agents.fact_checker_agent import FactCheckReport, make_fact_checker_node
from agents.writer_agent import make_writer_node
from graph.logger import wrap_for_logging
from tests.fakes import FakeLLM, fake_search


def _base_state(**overrides) -> ResearchState:
    """A fully-populated state dict; override any field with keyword args."""
    state: ResearchState = {
        "original_query": "Who created LangGraph?",
        "subtasks": [],
        "raw_search_results": {},
        "summarized_notes": {},
        "fact_check_flags": [],
        "checked_subtasks": [],
        "current_subtask": "",
        "final_answer": "",
        "current_step": "",
    }
    state.update(overrides)
    return state


# ── State schema ─────────────────────────────────────────────────────────────

class TestStateSchema:
    def test_state_has_all_expected_fields(self):
        s = _base_state()
        assert s["original_query"] == "Who created LangGraph?"
        assert s["subtasks"] == []
        assert s["raw_search_results"] == {}
        assert s["summarized_notes"] == {}
        assert s["fact_check_flags"] == []
        assert s["final_answer"] == ""

    def test_search_result_typed_dict_shape(self):
        r: SearchResult = {"title": "t", "snippet": "s", "url": "http://u"}
        assert r["title"] == "t" and r["snippet"] == "s" and r["url"] == "http://u"


# ── Search tool ──────────────────────────────────────────────────────────────

class TestSearchTool:
    def test_format_results_normalises_provider_keys(self):
        raw = [{"title": "T", "body": "B", "href": "http://t"}]
        out = search_tool._format_results(raw)
        assert out == [{"title": "T", "snippet": "B", "url": "http://t"}]

    def test_format_results_falls_back_for_missing_keys(self):
        raw = [{"title": "T", "body": "", "href": None}]
        out = search_tool._format_results(raw)
        assert out[0]["snippet"] == "" and out[0]["url"] == ""

    def test_web_search_returns_empty_list_on_provider_error(self, monkeypatch):
        class BoomDDGS:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                raise RuntimeError("network down")

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(search_tool, "DDGS", BoomDDGS)
        assert search_tool.web_search("anything") == []


# ── Orchestrator ─────────────────────────────────────────────────────────────

class TestOrchestrator:
    def test_decomposes_query_into_subtasks(self):
        node = make_orchestrator_node(FakeLLM())
        state = _base_state()
        updates = node(state)
        assert updates["subtasks"] == ["What is LangGraph?", "Who created LangGraph?"]
        assert updates["current_subtask"] == "What is LangGraph?"

    def test_skips_decomposition_when_subtasks_exist(self):
        llm = FakeLLM(subtasks=["Unexpected sub-task"])
        node = make_orchestrator_node(llm)
        state = _base_state(subtasks=["Existing task"])
        updates = node(state)
        # Must NOT re-decompose, and must hand out the existing pending task.
        assert "subtasks" not in updates
        assert updates["current_subtask"] == "Existing task"

    def test_dispatch_routes_to_search_when_task_pending(self):
        assert dispatch(_base_state(current_subtask="Q?")) == NODE_SEARCH

    def test_dispatch_routes_to_writer_when_all_done(self):
        assert dispatch(_base_state(current_subtask="")) == NODE_WRITER


# ── Search agent ─────────────────────────────────────────────────────────────

class TestSearchAgent:
    def test_stores_results_keyed_by_current_subtask(self):
        node = make_search_node(fake_search)
        updates = node(_base_state(current_subtask="Q"))
        assert updates["raw_search_results"]["Q"][0]["url"].startswith("https://")
        assert updates["current_step"] == "search"

    def test_stores_empty_list_key_when_no_results(self):
        node = make_search_node(lambda query, max_results=5: [])
        updates = node(_base_state(current_subtask="Q"))
        assert updates["raw_search_results"] == {"Q": []}

    def test_is_safe_without_current_subtask(self):
        node = make_search_node(fake_search)
        updates = node(_base_state(current_subtask=""))
        assert updates["current_step"] == "search"


# ── Summarizer ───────────────────────────────────────────────────────────────

class TestSummarizer:
    def test_produces_note_keyed_by_subtask(self):
        node = make_summarizer_node(FakeLLM(note_text="tiny note"))
        state = _base_state(
            current_subtask="Q",
            raw_search_results={"Q": [{"title": "t", "snippet": "s", "url": "http://x"}]},
        )
        updates = node(state)
        assert updates["summarized_notes"] == {"Q": "tiny note"}
        assert updates["current_step"] == "summarizer"

    def test_summarizes_empty_results_as_no_information(self):
        class EchoLLM:
            def invoke(self, messages):
                text = " ".join(m.content for m in messages if hasattr(m, "content"))
                return type("M", (), {"content": text})()

        node = make_summarizer_node(EchoLLM())
        state = _base_state(current_subtask="Q", raw_search_results={"Q": []})
        updates = node(state)
        assert "no results returned" in updates["summarized_notes"]["Q"]


# ── Fact-Checker ─────────────────────────────────────────────────────────────

class TestFactChecker:
    def test_flags_are_appended_and_subtask_marked_checked(self):
        node = make_fact_checker_node(FakeLLM(flags=["claim overstated"]))
        state = _base_state(
            current_subtask="Q",
            summarized_notes={"Q": "some note"},
            raw_search_results={"Q": [{"title": "t", "snippet": "s", "url": "http://x"}]},
            fact_check_flags=["pre-existing flag"],
            checked_subtasks=["Q0"],
        )
        updates = node(state)
        assert updates["fact_check_flags"] == ["claim overstated"]
        assert updates["checked_subtasks"] == ["Q"]
        assert updates["current_step"] == "fact_checker"

    def test_structed_output_report_defaults(self):
        r = FactCheckReport(flags=[], report="all good")
        assert r.flags == [] and r.report == "all good"


# ── Writer ───────────────────────────────────────────────────────────────────

class TestWriter:
    def test_assembles_final_answer_with_notes_and_flags(self):
        # Echo the prompt so we can assert what the writer received.
        class EchoLLM:
            def invoke(self, messages):
                text = " ".join(m.content for m in messages if hasattr(m, "content"))
                return type("M", (), {"content": text})()

        node = make_writer_node(EchoLLM())
        state = _base_state(
            original_query="What is LangGraph?",
            summarized_notes={"Q1": "Note one", "Q2": "Note two"},
            fact_check_flags=["Q1 unsupported"],
        )
        updates = node(state)
        assert "What is LangGraph?" in updates["final_answer"]
        assert "Note one" in updates["final_answer"]
        assert "Q1 unsupported" in updates["final_answer"]
        assert updates["current_step"] == "writer"


# ── Logger wrapper ───────────────────────────────────────────────────────────

class TestLoggerWrapper:
    def test_wrapper_preserves_node_behaviour(self):
        def fake_node(state):
            return {"current_step": "made_up"}

        wrapped = wrap_for_logging("made_up", fake_node)
        assert wrapped(_base_state()) == {"current_step": "made_up"}