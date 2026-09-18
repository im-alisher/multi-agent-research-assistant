"""
evaluation/collector.py — Turns the aftermath of a run into ``RunMetrics``.

This is the "measurement" stage of the evaluation framework. It answers the
question *"what actually happened during this run?"* by reading two artifacts
that the existing system already produces, without touching any agent logic:

* the **final shared state** returned by ``graph.invoke(...)`` (sub-tasks,
  raw results keyed by sub-task, summaries, fact-check flags, final answer),
* the **structured log entries** recorded by ``graph/logger.py`` (one
  "received" + one "produced" entry per node visit, with timestamps).

Because everything is derived from what the run *already* left behind, the
collector can never change the run itself — a nice property when you want to
retrospectively evaluate runs that were executed before this framework existed.

Key design choices
------------------
1. **Log entries are the clock.** ``total_run_time_s`` is the difference
   between the first and the last captured entry's timestamp. Nothing else can
   tell us how long the loop took without re-timing it.
2. **Node transitions come from the "received" lines.** Each time the graph
   enters a node the wrapper logs a ``>> agent: ... received:`` line; that is
   one transition. Produced lines are not counted, so one node visit == one
   transition.
3. **Flag-to-sub-task attribution is done by correlation, not parsing.** The
   Fact-Checker visits sub-tasks in the same order they are appended to
   ``checked_subtasks``, and its log line records how many flags *that visit*
   produced. Pairing the two lists recovers ``flags_by_subtask`` even though
   the flag strings themselves are not tagged with a sub-task (that would
   require changing the agent — exactly what we are avoiding). See
   ``_flags_by_subtask``.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

from evaluation.metrics import RunMetrics

# -- Log entry kinds (mirrors graph/logger.py's two line prefixes) -----------
KIND_RECEIVED = "received"   # ">> agent: <name> | ... received: ..."
KIND_PRODUCED = "produced"   # "   agent: <name> | ... produced: ..."

# A summary note shorter than this many characters is considered trivial —
# it cannot possibly carry real information for the Writer to reuse.
TRIVIAL_SUMMARY_MIN_LEN = 20

# Words shorter than this are ignored when checking whether the final answer
# "mentions" a sub-task (articles, prepositions, etc. carry no signal).
SIGNIFICANT_WORD_MIN_LEN = 3

_RECEIVED_RE = re.compile(r"^>>\s*agent:\s*(\S+)\s*\|")
_PRODUCED_RE = re.compile(r"^\s*agent:\s*(\S+)\s*\|\s*produced:")
_FLAG_COUNT_RE = re.compile(r"fact_check_flags=\[(\d+) item\(s\)\]")


@dataclass(frozen=True)
class LogEntry:
    """
    One structured entry derived from a raw ``agent_comm`` log line.

    Attributes:
        agent: The node name (e.g. ``"search"``, ``"fact_checker"``).
        kind: ``KIND_RECEIVED`` (the graph entered this node) or
            ``KIND_PRODUCED`` (the node returned its state update).
        timestamp: Epoch seconds when the record was created (float precision).
        message: The original log line text, kept for debugging and for the
            small amount of parsing the collector needs (see ``_flags_by_subtask``).
    """

    agent: str
    kind: str
    timestamp: float
    message: str


class LogRecordCollector(logging.Handler):
    """
    A logging handler that quietly accumulates every :class:`logging.LogRecord`
    it receives and never writes anything anywhere.

    The point is isolation of a *single run*: ``main.py`` attaches one of these
    to the ``"agent_comm"`` logger for the duration of one ``graph.invoke()``,
    then detaches it. Anything captured therefore belongs to exactly that run,
    unlike the shared ``logs/agent_comm.log`` file which accumulates across
    every run. This is how the evaluation framework reads "this run's logs"
    without modifying the existing file logger or any agent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def capture_agent_log(
    logger: logging.Logger,
) -> Iterator[LogRecordCollector]:
    """
    Context manager that attaches a :class:`LogRecordCollector` to ``logger``
    for its duration and always detaches it again (even on exceptions).

    Usage::

        logger = get_logger()          # the "agent_comm" logger (module-level)
        with capture_agent_log(logger) as collector:
            final_state = graph.invoke(state)
        entries = entries_from_records(collector.records)

    Args:
        logger: The logger instance to tap (usually ``graph.logger.get_logger()``).

    Yields:
        The collector, whose ``.records`` list is populated while the context is open.
    """
    collector = LogRecordCollector()
    logger.addHandler(collector)
    try:
        yield collector
    finally:
        logger.removeHandler(collector)


def _parse_agent_and_kind(message: str) -> tuple[str, str] | None:
    """
    Turn one raw log line into ``(agent, kind)``.

    Lines produced by ``graph/logger.py`` come in only two shapes:

        * ``>> agent: <name> | ... received: ...``  -> entered a node
        * ``   agent: <name> | ... produced: ...``  -> node returned an update

    Returns ``None`` for anything that matches neither shape (foreign lines
    from other loggers, malformed output, ...) — those are silently skipped
    rather than crashing a whole evaluation.
    """
    received = _RECEIVED_RE.match(message.strip())
    if received:
        return received.group(1), KIND_RECEIVED
    produced = _PRODUCED_RE.match(message)
    if produced:
        return produced.group(1), KIND_PRODUCED
    return None


def entries_from_records(
    records: Sequence[logging.LogRecord],
) -> list[LogEntry]:
    """
    Convert captured ``logging.LogRecord`` objects into a list of ``LogEntry``.

    Args:
        records: Log records collected by a :class:`LogRecordCollector`.

    Returns:
        Structured entries; non-agent-communication records are dropped.
    """
    entries: list[LogEntry] = []
    for record in records:
        message = record.getMessage()
        parsed = _parse_agent_and_kind(message)
        if parsed is None:
            continue
        agent, kind = parsed
        entries.append(LogEntry(agent=agent, kind=kind, timestamp=record.created, message=message))
    return entries


def _subtasks(state: dict) -> list[str]:
    """Safe accessor for the sub-task list in a (possibly partial) state."""
    return list(state.get("subtasks") or [])


def _raw_text_length(results: list[dict]) -> int:
    """
    Total length (in characters) of the raw search material for one sub-task.

    Used by the summarizer's "did it actually condense?" check. Counting
    titles + snippets approximates "how much text was fed in", so a shorter
    note demonstrates genuine condensation rather than echo.
    """
    return sum(
        len(str(r.get("title", ""))) + len(str(r.get("snippet", "")))
        for r in results
    )


def _flag_counts_from_log(entries: Sequence[LogEntry]) -> list[int]:
    """
    Recover how many flags *each* Fact-Checker visit produced.

    The log wrapper summarises a Fact-Checker's returned update as
    ``produced: {..., fact_check_flags=[N item(s)], ...}``. Extracting ``N``
    from every Fact-Checker "produced" line, in order, gives one number per
    visit — which is exactly what we need to attribute flags to sub-tasks.
    """
    counts: list[int] = []
    for entry in entries:
        if entry.agent != "fact_checker" or entry.kind != KIND_PRODUCED:
            continue
        match = _FLAG_COUNT_RE.search(entry.message)
        if match:
            counts.append(int(match.group(1)))
    return counts


def _flags_by_subtask(state: dict, entries: Sequence[LogEntry]) -> dict[str, int]:
    """
    Attribute fact-check flags to the sub-task they were raised on.

    Why this works: the Fact-Checker appends to ``checked_subtasks`` once per
    visit, in visit order, and the log records how many flags each visit
    produced. Pairing the i-th checked sub-task with the i-th flag count
    therefore reconstructs per-sub-task attribution *without* the flags being
    tagged (they never are — the agent writes plain strings).

    If the pairing breaks down (missing/partial logs), we degrade gracefully:
    any sub-task we cannot attribute a count to simply defaults to 0.
    """
    checked = list(state.get("checked_subtasks") or [])
    counts = _flag_counts_from_log(entries)
    attributed: dict[str, int] = {}
    for index, subtask in enumerate(checked):
        attributed[subtask] = counts[index] if index < len(counts) else 0
    return attributed


def _mentions_answer(answer: str, subtask: str) -> bool:
    """
    Heuristic: did the final answer actually reference ``subtask``?

    The check is deliberately tolerant because the Writer is an LLM that may
    paraphrase a sub-task heading. We accept the sub-task as "covered" if:
    (1) the exact sub-task text appears in the answer (case-insensitive), or
    (2) every *significant* word of the sub-task appears somewhere in the
    answer (so "Will AI agents replace software engineers?" counts as covered
    even when phrased differently).

    This is a heuristic, not a proof: it can miss a fluent paraphrase that
    avoids the sub-task's own wording. For a learning project that trade-off
    (simple, explainable, zero model calls) beats accuracy.
    """
    if not subtask or not answer:
        return False
    haystack = answer.lower()
    if subtask.lower() in haystack:
        return True
    significant = re.findall(r"[a-z0-9]{%d,}" % SIGNIFICANT_WORD_MIN_LEN, subtask.lower())
    return bool(significant) and all(word in haystack for word in significant)


def collect(final_state: dict, log_entries: Sequence[LogEntry]) -> RunMetrics:
    """
    Compute a fully-populated :class:`RunMetrics` from the final run state
    and the structured log entries captured for that run.

    Args:
        final_state: The state dict returned by ``graph.invoke(...)``. On an
            errored run this may be the partial state (e.g. the initial state
            if the graph raised before completing) — the collector degrades
            gracefully and merely reports ``completed_successfully=False``.
        log_entries: Structured :class:`LogEntry` objects for the run, in the
            order the records were captured.

    Returns:
        A :class:`RunMetrics` instance. It contains only *measurements*; no
        judgement happens here (that is :mod:`evaluation.scorer`'s job).

    The function is pure: it reads, it sums, it counts. It never mutates
    ``final_state`` or the entries, so it is trivially testable.
    """
    subtasks = _subtasks(final_state)
    raw = final_state.get("raw_search_results") or {}
    notes = final_state.get("summarized_notes") or {}
    answer = str(final_state.get("final_answer") or "")

    # ── Overall ──────────────────────────────────────────────────────────────
    timestamps = [entry.timestamp for entry in log_entries if entry.timestamp]
    run_time = 0.0
    if timestamps:
        run_time = max(0.0, max(timestamps) - min(timestamps))

    completed_successfully = (
        final_state.get("current_step") == "writer" and bool(answer)
    )

    # ── Orchestrator ─────────────────────────────────────────────────────────
    checked = final_state.get("checked_subtasks") or []
    loop_terminated = set(checked) >= set(subtasks)
    # A run that never produced sub-tasks trivially "terminated"; only flag non-
    # termination when there WAS something to process that never got checked.

    # ── Search ───────────────────────────────────────────────────────────────
    search_calls = sum(
        1 for entry in log_entries
        if entry.agent == "search" and entry.kind == KIND_RECEIVED
    )
    empty_results = [st for st in subtasks if not raw.get(st)]

    # ── Summarizer ───────────────────────────────────────────────────────────
    summarized = sum(1 for st in subtasks if st in notes)
    not_condensed: list[str] = []
    trivial: list[str] = []
    for st in subtasks:
        note = notes.get(st)
        if note is None:
            continue
        raw_len = _raw_text_length(raw.get(st) or [])
        # Condensation is only meaningful when there was material to condense.
        if raw_len > 0 and len(note) >= raw_len:
            not_condensed.append(st)
        if len(str(note).strip()) < TRIVIAL_SUMMARY_MIN_LEN:
            trivial.append(st)

    # ── Fact-checker ─────────────────────────────────────────────────────────
    flags_total = len(final_state.get("fact_check_flags") or [])
    flags_by_subtask = _flags_by_subtask(final_state, log_entries)

    # ── Writer ───────────────────────────────────────────────────────────────
    not_covered = [st for st in subtasks if not _mentions_answer(answer, st)]

    return RunMetrics(
        original_query=str(final_state.get("original_query") or ""),
        completed_successfully=completed_successfully,
        total_run_time_s=run_time,
        total_node_transitions=sum(
            1 for entry in log_entries if entry.kind == KIND_RECEIVED
        ),
        orchestrator_subtask_count=len(subtasks),
        orchestrator_loop_terminated=loop_terminated,
        search_calls=search_calls,
        subtasks_with_empty_results=empty_results,
        subtasks_summarized=summarized,
        summaries_not_condensed=not_condensed,
        trivial_summaries=trivial,
        fact_check_flags_total=flags_total,
        flags_by_subtask=flags_by_subtask,
        answer_length=len(answer),
        subtasks_not_covered=not_covered,
    )