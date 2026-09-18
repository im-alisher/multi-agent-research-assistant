"""
evaluation/metrics.py — The run-level metrics data model.

A research run produces two observable things:

1. ``ResearchState`` — the final shared state left behind after the graph
   finishes (subtasks, raw results, notes, flags, final answer, ...).
2. A stream of structured log entries — one "received" + one "produced" line
   per node entry, recorded by ``graph/logger.py``.

The whole evaluation framework is a pipeline over those two artifacts:

    final state + log entries
        ──collect──▶ RunMetrics     (numeric facts about the run)
        ──score────▶ Verdicts       (pass / warning per agent)
        ──report──▶ text           (human-readable evaluation report)

``RunMetrics`` is the first link in that chain: a plain, self-describing value
object that holds every number the later stages need, grouped by agent. Splitting
"measure" (collector) from "judge" (scorer) is a deliberate design choice: the
same numbers can be judged with different rule sets later without re-measuring,
which is exactly the shape you want if you ever graduate this toy framework into
something a professor would accept as a "metric harness".

Why a dataclass and not Pydantic? Pydantic already exists in this project (the
agents use it for structured LLM output), but ``RunMetrics`` is not user input —
it is computed data we construct ourselves. A dataclass keeps construction
cheap and dependency-light, which makes hand-crafting instances in tests
trivial. The trade-off (no runtime validation) is fine because the collector is
the only producer and it is kept simple on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    """
    One instance of this class captures every measured fact about a single run.

    Fields are grouped below in the same order as the evaluation goals in the
    README: Orchestrator, Search, Summarizer, Fact-Checker, Writer, Overall.

    Design note — *facts, not verdicts*: every field here is a raw measurement
    (a count, a length, a list of strings). Whether any of them is "good" or
    "bad" is deliberately NOT decided here; that is the scorer's job
    (``evaluation/scorer.py``). Keeping measurement and judgement separate is
    what makes the scoring rules replaceable without touching this class.
    """

    # ── Overall / run bookkeeping ────────────────────────────────────────────
    original_query: str = ""
    """The question the user asked. Purely for readability in reports."""

    completed_successfully: bool = False
    """
    Whether the graph reached the Writer node and produced an answer.
    Derived by the collector from the final state (``current_step == "writer"``
    and a non-empty ``final_answer``). A run that errored out mid-flight will
    be ``False`` and the scorer turns that into a warning.
    """

    total_run_time_s: float = 0.0
    """
    Wall-clock seconds the graph loop took, derived by the collector as the
    difference between the first and the last log entry timestamps. This
    excludes graph build time, on purpose: the decision "did this run get
    stuck?" should be about the loop itself, not about LLM instantiation.
    """

    total_node_transitions: int = 0
    """
    Number of times the graph entered a node. Each captured "received" log
    entry is one transition. For a healthy 3-sub-task run you expect roughly:
    1 (orchestrator decomposition) + 3×3 (search/summarize/fact-check per
    sub-task) + 1 (orchestrator hand-outs) + 1 (writer) ≈ 12. If this number
    is identical to the sub-task count you can already smell an infinite loop.
    """

    # ── Orchestrator ─────────────────────────────────────────────────────────
    orchestrator_subtask_count: int = 0
    """How many sub-tasks the Orchestrator decomposed the query into."""

    orchestrator_loop_terminated: bool = True
    """
    ``True`` when every sub-task was marked "checked" (i.e. the
    ``checked_subtasks`` set covers the ``subtasks`` set). If this is ``False``
    the router would never have found a done state, which is exactly how an
    infinite / stuck loop manifests. (If there were zero sub-tasks the loop
    trivially terminated.)
    """

    # ── Search ──────────────────────────────────────────────────────────────
    search_calls: int = 0
    """Total number of web searches issued during the run (one per node visit)."""

    results_by_subtask: dict[str, int] = field(default_factory=dict)
    """
    How many raw search results each sub-task yielded, keyed by sub-task text.
    Included so the report can show the result counts at a glance (and so the
    scorer/report never needs the raw state again). A missing key means 0.
    """

    subtasks_with_empty_results: list[str] = field(default_factory=list)
    """
    Sub-tasks for which the Search agent returned zero results. Retrieved from
    ``raw_search_results``; a sub-task with no key at all (never searched)
    is treated the same as an explicitly empty result list.
    """

    # ── Summarizer ───────────────────────────────────────────────────────────
    subtasks_summarized: int = 0
    """How many sub-tasks got a summary note in ``summarized_notes``."""

    summaries_not_condensed: list[str] = field(default_factory=list)
    """
    Sub-tasks whose summary note is as long as (or longer than) the raw search
    text it was built from. A summarizer that fails this check is just copying
    — it is not doing the "condense" half of its job.
    """

    trivial_summaries: list[str] = field(default_factory=list)
    """
    Sub-tasks whose summary is empty or near-empty placeholder text (say,
    shorter than MIN_SUMMARY_LENGTH characters). A trivial summary adds no
    information for the Writer to draw on.
    """

    # ── Fact-checker ─────────────────────────────────────────────────────────
    fact_check_flags_total: int = 0
    """Total number of warning flags raised across all sub-tasks."""

    flags_by_subtask: dict[str, int] = field(default_factory=dict)
    """
    Flag count keyed by sub-task. Built by the collector from the run order:
    the Fact-Checker visits sub-tasks in the same order they were "checked",
    and the log records how many flags each visit produced, so the two can be
    correlated deterministically.
    """

    # ── Writer ───────────────────────────────────────────────────────────────
    answer_length: int = 0
    """
    ``len(final_answer)`` in characters. The scorer uses this against a
    documented MIN/MAX to catch obviously truncated or bloated answers.
    """

    subtasks_not_covered: list[str] = field(default_factory=list)
    """
    Sub-tasks whose text (or enough of its significant words) does not appear
    anywhere in the final answer. A Writer that silently drops a sub-task has
    failed to "answer the whole question".
    """

    # ── Convenience: derived lookups ─────────────────────────────────────────
    @property
    def subtasks_with_results(self) -> int:
        """Number of sub-tasks that yielded at least one search result."""
        return self.orchestrator_subtask_count - len(self.subtasks_with_empty_results)

    @property
    def fact_check_flags_ratio(self) -> float:
        """
        Fraction of sub-tasks that had at least one flag raised.
        Guarded against zero sub-tasks so the scorer never divides by zero.
        """
        if self.orchestrator_subtask_count == 0:
            return 0.0
        flagged = sum(1 for count in self.flags_by_subtask.values() if count > 0)
        return flagged / self.orchestrator_subtask_count

    @property
    def any_trivial_or_uncondensed_summaries(self) -> bool:
        """Convenience flag: a summarizer had *some* quality problem."""
        return bool(self.trivial_summaries or self.summaries_not_condensed)