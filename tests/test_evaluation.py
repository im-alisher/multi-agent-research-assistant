"""
tests/test_evaluation.py — Unit tests for the agent evaluation framework.

Three layers are exercised, mirroring the framework's pipeline:

* ``entries_from_records`` — regex parsing of raw agent-communication
  ``logging.LogRecord`` objects into structured ``LogEntry`` values.
* ``collect`` (the collector) — a FAKE completed run's state + a hand-crafted
  log-entry timeline, asserting that every ``RunMetrics`` field is correctly
  derived (search call counts, flag attribution per sub-task, run time from
  log timestamps, loop-termination, writer coverage, ...). Also covers the
  partial/errored-state case.
* ``score`` (the scorer) — hand-built ``RunMetrics`` instances engineered to
  trip specific rules, asserting the exact PASS/WARN verdict per agent.

None of these tests touches the live Groq API, the web, or the real graph —
everything is hand-crafted dictionaries and dataclasses.
"""

from __future__ import annotations

import logging

import pytest

from evaluation.collector import (
    KIND_PRODUCED,
    KIND_RECEIVED,
    LogEntry,
    collect,
    entries_from_records,
)
from evaluation.metrics import RunMetrics
from evaluation.scorer import score

SUB_A = "What are LangGraph's capabilities?"
SUB_B = "How does LangGraph route agents?"


# ── Helpers: fake state and fake log timeline for a completed 2-sub-task run ─

def _result_list(n: int) -> list[dict]:
    """A fake pile of ``n`` raw search results for a sub-task."""
    return [
        {"title": f"Title {i}" * 10, "snippet": f"Snippet {i}" * 10, "url": f"https://x/{i}"}
        for i in range(n)
    ]


def _completed_state() -> dict:
    """A fully finished state, as ``graph.invoke`` would leave it."""
    final_answer = (
        f"The full answer considers both aspects. {SUB_A} "
        f"Meanwhile routing is covered by {SUB_B}. "
        "This sentence adds length so the answer clears minimum bounds. "
    ) * 3
    return {
        "original_query": "Explain the LangGraph architecture",
        "subtasks": [SUB_A, SUB_B],
        "raw_search_results": {SUB_A: _result_list(1), SUB_B: _result_list(2)},
        "summarized_notes": {
            SUB_A: "Concise note about capabilities. Sources: http://a",
            SUB_B: "Concise note about routing. Sources: http://b",
        },
        "current_subtask": "",
        "checked_subtasks": [SUB_A, SUB_B],
        # One flag per sub-task, appended in the same order sub-tasks were
        # checked — exactly what the real Fact-Checker does.
        "fact_check_flags": ["capabilities overstated", "routing claim unsupported"],
        "final_answer": final_answer,
        "current_step": "writer",
    }


def _completed_log_entries() -> list[LogEntry]:
    """
    Timeline of a healthy 2-sub-task run, as ``graph/logger.py`` would emit it.

    Structure per sub-task: orchestrator (hand-out) -> search -> summarizer ->
    fact_checker, then one final orchestrator hand-out (none left) -> writer.
    Processing order is [SUB_A, SUB_B]. The fact_checker "produced" lines carry
    ``fact_check_flags=[N item(s)]``, which the collector parses to attribute
    the 2 flags (1 per sub-task). Timestamps step 1s apart from 100.0, so run
    time = last - first = 21.0s and received-lines = 11 transitions
    (3 orchestrator visits: decompose + 2 hand-outs - plus search/summarizer/
    fact_checker/ writer, i.e. 11 node entries in total).
    """
    t = 100.0
    entries: list[LogEntry] = []

    def received(agent: str, note: str = "") -> None:
        nonlocal t
        entries.append(LogEntry(agent, KIND_RECEIVED, t, f">> agent: {agent} | received: {note}"))
        t += 1.0

    def produced(agent: str, note: str = "") -> None:
        nonlocal t
        entries.append(LogEntry(agent, KIND_PRODUCED, t, f"   agent: {agent} | produced: {note}"))
        t += 1.0

    # Entry 1: decomposition.
    received("orchestrator", "current_step=")
    produced("orchestrator", "{subtasks=[2 item(s)], current_subtask=" + SUB_A + "}")

    # SUB_A pipeline.
    received("orchestrator", "current_step=orchestrator")
    produced("orchestrator", "{current_subtask=" + SUB_A + "}")
    received("search", "current_step=orchestrator")
    produced("search", "{current_step=search, raw_search_results={" + SUB_A + ":[1 item(s)]}}")
    received("summarizer", "current_step=search")
    produced("summarizer", "{summarized_notes={" + SUB_A + "=Concise note}}")
    received("fact_checker", "current_step=summarizer")
    produced("fact_checker", "{fact_check_flags=[1 item(s)], checked_subtasks=[1 item(s)]}")

    # SUB_B pipeline.
    received("orchestrator", "current_step=fact_checker")
    produced("orchestrator", "{current_subtask=" + SUB_B + "}")
    received("search", "current_step=orchestrator")
    produced("search", "{current_step=search, raw_search_results={" + SUB_B + ":[2 item(s)]}}")
    received("summarizer", "current_step=search")
    produced("summarizer", "{summarized_notes={" + SUB_B + "=Concise note}}")
    received("fact_checker", "current_step=summarizer")
    produced("fact_checker", "{fact_check_flags=[1 item(s)], checked_subtasks=[1 item(s)]}")

    # No sub-tasks left -> writer.
    received("orchestrator", "current_step=fact_checker")
    produced("orchestrator", "{current_subtask=}")
    received("writer", "current_step=orchestrator")
    produced("writer", "{current_step=writer, final_answer=The final answer}}")

    return entries


def _healthy_metrics(**overrides) -> RunMetrics:
    """A metrics object where every agent would pass; override per test."""
    m = RunMetrics(
        original_query="q",
        completed_successfully=True,
        total_run_time_s=3.0,
        total_node_transitions=12,
        orchestrator_subtask_count=3,
        orchestrator_loop_terminated=True,
        search_calls=3,
        results_by_subtask={f"s{i}": 1 for i in range(3)},
        subtasks_summarized=3,
        fact_check_flags_total=1,
        flags_by_subtask={"s0": 1, "s1": 0, "s2": 0},
        answer_length=1500,
        subtasks_not_covered=[],
    )
    m.__dict__.update(overrides)
    return m


def _verdict_for(metrics: RunMetrics, agent: str):
    """Return the single verdict object for ``agent``."""
    return next(v for v in score(metrics) if v.agent == agent)


# ── Log parsing ──────────────────────────────────────────────────────────────

class TestEntriesFromRecords:
    def test_parses_received_and_produced_lines(self):
        rec = logging.LogRecord("agent_comm", logging.INFO, __file__, 1,
                                ">> agent: search | received: {x}", None, None)
        rec.created = 1.0
        rec2 = logging.LogRecord("agent_comm", logging.INFO, __file__, 1,
                                 "   agent: fact_checker | produced: {y}", None, None)
        rec2.created = 2.0
        entries = entries_from_records([rec, rec2])
        assert entries == [
            LogEntry("search", KIND_RECEIVED, 1.0, ">> agent: search | received: {x}"),
            LogEntry("fact_checker", KIND_PRODUCED, 2.0, "   agent: fact_checker | produced: {y}"),
        ]

    def test_drops_foreign_log_lines(self):
        rec = logging.LogRecord("other", logging.INFO, __file__, 1,
                                "garbage from another logger", None, None)
        rec.created = 1.0
        assert entries_from_records([rec]) == []


# ── Collector ────────────────────────────────────────────────────────────────

class TestCollect:
    def test_completed_run_produces_full_metrics(self):
        entries = _completed_log_entries()
        metrics = collect(_completed_state(), entries)

        # Overall
        assert metrics.completed_successfully is True
        assert metrics.total_node_transitions == 11
        assert metrics.total_run_time_s == pytest.approx(21.0)

        # Orchestrator
        assert metrics.orchestrator_subtask_count == 2
        assert metrics.orchestrator_loop_terminated is True

        # Search
        assert metrics.search_calls == 2
        assert metrics.results_by_subtask == {SUB_A: 1, SUB_B: 2}
        assert metrics.subtasks_with_empty_results == []

        # Summarizer
        assert metrics.subtasks_summarized == 2
        assert metrics.summaries_not_condensed == []
        assert metrics.trivial_summaries == []

        # Fact-checker: 2 flags, attributed 1 per sub-task via log correlation.
        assert metrics.fact_check_flags_total == 2
        assert metrics.flags_by_subtask == {SUB_A: 1, SUB_B: 1}

        # Writer: answer long enough and mentions both sub-tasks.
        assert metrics.answer_length == len(_completed_state()["final_answer"])
        assert metrics.subtasks_not_covered == []

    def test_empty_search_results_are_reported(self):
        state = _completed_state()
        state["raw_search_results"][SUB_B] = []  # SUB_B found nothing
        metrics = collect(state, _completed_log_entries())
        assert metrics.subtasks_with_empty_results == [SUB_B]

    def test_subtask_missing_from_results_is_treated_as_empty(self):
        state = _completed_state()
        del state["raw_search_results"][SUB_A]  # never even searched
        metrics = collect(state, _completed_log_entries())
        assert metrics.subtasks_with_empty_results == [SUB_A]
        assert metrics.results_by_subtask[SUB_A] == 0

    def test_errored_partial_state_is_not_successful(self):
        # A run that died mid-flight: orchestrator decomposed, search ran for
        # the first sub-task, then something threw before anything completed.
        state = {
            "original_query": "q",
            "subtasks": [SUB_A, SUB_B],
            "raw_search_results": {SUB_A: _result_list(1)},
            "summarized_notes": {},
            "current_subtask": SUB_A,
            "checked_subtasks": [],
            "fact_check_flags": [],
            "final_answer": "",
            "current_step": "search",
        }
        entries = [
            LogEntry("orchestrator", KIND_RECEIVED, 100.0, ">> agent: orchestrator | x"),
            LogEntry("orchestrator", KIND_PRODUCED, 101.0, "   agent: orchestrator | y"),
            LogEntry("search", KIND_RECEIVED, 102.0, ">> agent: search | x"),
            LogEntry("search", KIND_PRODUCED, 103.0, "   agent: search | y"),
        ]
        metrics = collect(state, entries)

        assert metrics.completed_successfully is False
        assert metrics.orchestrator_loop_terminated is False  # B never checked
        assert metrics.search_calls == 1
        assert metrics.fact_check_flags_total == 0
        assert metrics.subtasks_not_covered == [SUB_A, SUB_B]  # no answer at all
        assert metrics.total_run_time_s == pytest.approx(3.0)  # 100 -> 103


# ── Scorer ───────────────────────────────────────────────────────────────────

class TestScorerOrchestrator:
    def test_pass_when_within_2_to_4(self):
        assert _verdict_for(_healthy_metrics(), "orchestrator").status == "pass"

    def test_warning_when_only_one_subtask(self):
        m = _healthy_metrics(orchestrator_subtask_count=1)
        v = _verdict_for(m, "orchestrator")
        assert v.status == "warning" and "expected 2-4" in v.explanation

    def test_warning_when_five_or_more(self):
        m = _healthy_metrics(orchestrator_subtask_count=5)
        assert _verdict_for(m, "orchestrator").status == "warning"

    def test_warning_when_loop_did_not_terminate(self):
        m = _healthy_metrics(orchestrator_loop_terminated=False)
        v = _verdict_for(m, "orchestrator")
        assert v.status == "warning" and "did not mark every sub-task" in v.explanation


class TestScorerSearch:
    def test_pass_when_all_subtasks_yielded_results(self):
        assert _verdict_for(_healthy_metrics(), "search").status == "pass"

    def test_warning_when_some_subtask_returned_nothing(self):
        m = _healthy_metrics(subtasks_with_empty_results=["s1"])
        v = _verdict_for(m, "search")
        assert v.status == "warning" and "s1" in v.explanation


class TestScorerSummarizer:
    def test_pass_when_all_condensed_and_non_trivial(self):
        assert _verdict_for(_healthy_metrics(), "summarizer").status == "pass"

    def test_warning_when_a_summary_is_trivial(self):
        m = _healthy_metrics(trivial_summaries=["s2"])
        assert _verdict_for(m, "summarizer").status == "warning"

    def test_warning_when_a_summary_was_not_condensed(self):
        m = _healthy_metrics(summaries_not_condensed=["s0", "s1"])
        assert _verdict_for(m, "summarizer").status == "warning"

    def test_warning_when_not_every_subtask_was_summarised(self):
        m = _healthy_metrics(subtasks_summarized=2)
        assert _verdict_for(m, "summarizer").status == "warning"


class TestScorerFactChecker:
    def test_pass_when_flags_raise_on_at_most_half(self):
        assert _verdict_for(_healthy_metrics(), "fact_checker").status == "pass"

    def test_warning_when_flags_raise_on_more_than_half(self):
        # s0 and s1 flagged out of 3 sub-tasks -> ratio 67% > threshold.
        m = _healthy_metrics(
            fact_check_flags_total=2,
            flags_by_subtask={"s0": 1, "s1": 1, "s2": 0},
        )
        v = _verdict_for(m, "fact_checker")
        assert v.status == "warning"
        assert "67%" in v.explanation


class TestScorerWriter:
    def test_pass_when_answer_covers_all_and_reasonable_length(self):
        assert _verdict_for(_healthy_metrics(), "writer").status == "pass"

    def test_warning_when_a_subtask_is_not_mentioned(self):
        m = _healthy_metrics(subtasks_not_covered=["s1"])
        assert _verdict_for(m, "writer").status == "warning"

    def test_warning_when_answer_is_too_short(self):
        m = _healthy_metrics(answer_length=12)
        assert _verdict_for(m, "writer").status == "warning"

    def test_warning_when_answer_is_too_long(self):
        m = _healthy_metrics(answer_length=50_000)
        assert _verdict_for(m, "writer").status == "warning"


class TestScorerOverall:
    def test_pass_on_successful_healthy_run(self):
        assert _verdict_for(_healthy_metrics(), "overall").status == "pass"

    def test_warning_when_run_errored(self):
        m = _healthy_metrics(completed_successfully=False)
        assert _verdict_for(m, "overall").status == "warning"

    def test_warning_when_nothing_ran(self):
        m = _healthy_metrics(total_node_transitions=0)
        assert _verdict_for(m, "overall").status == "warning"

    def test_every_agent_receives_a_verdict(self):
        agents = {v.agent for v in score(_healthy_metrics())}
        assert agents == {"orchestrator", "search", "summarizer", "fact_checker", "writer", "overall"}