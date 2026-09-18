"""
evaluation/scorer.py — Turn ``RunMetrics`` into per-agent pass/warning verdicts.

This is the "judgement" stage. It is deliberately a tiny, transparent set of
hand-written rules — NOT a learned model, NOT a rubric-scored LLM callback —
for two reasons that matter for a learning project:

1. **Explainability.** Every line of this file is something you can read aloud
   in a sentence ("the Writer got a warning because its answer was 12
   characters long and the minimum is 100"). A professor can audit these rules
   in minutes; a learned scorer is a black box.
2. **Separability of measure vs. judge.** Because the collector (:mod:`collector`)
   stores raw *facts* and the scorer holds all *judgement*, swapping a rule
   here never requires re-running an experiment. If you later want a weighted
   rubric or a learned regressor, you replace ONLY this module.

Status vocabulary is deliberately binary-plus: ``pass`` or ``warning``. We stop
short of a numeric grade — a single 0-100 "agent quality" number would imply a
rigour we do not have (the measurements are heuristics). Pass/warning is honest
about the uncertainty while still being actionable.

The rules below each carry a ``# why`` comment quoting the exact design target
they enforce, so the mapping from metric → rule is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from evaluation.metrics import RunMetrics

# ── Rule thresholds ──────────────────────────────────────────────────────────
# These constants centralise every "magic number", so the rules are readable as
# data and trivially tuneable.

# Orchestrator: the decomposition prompt (agents/orchestrator.py) asks for
# "between 2 and {max_subtasks}" where max=4. Anything outside that window means
# the model ignored its instructions (and 1 huge sub-task defeats decomposition,
# while 5+ fragments the answer beyond readability).
ORCH_MIN_SUBTASKS = 2
ORCH_MAX_SUBTASKS = 4

# Writer: the prompt says "aim for well under a page". As characters: 100 is a
# generous floor below which the answer cannot have said anything useful about
# 2+ sub-tasks, and 20_000 is a generous ceiling above which the Writer ignored
# the "keep it concise" instruction.
ANSWER_MIN_LEN = 100
ANSWER_MAX_LEN = 20_000

# Fact-checker: if MORE than half of the sub-tasks got at least one flag, the
# upstream summaries are systematically shaky — that is worth a closer look
# (an informational warning, since flag-count alone doesn't prove quality).
FACT_CHECK_RATIO_THRESHOLD = 0.5


@dataclass(frozen=True)
class Verdict:
    """
    One agent's evaluation outcome.

    Attributes:
        agent: Which agent (``"orchestrator"``, ``"search"``, ...).
        status: ``"pass"`` or ``"warning"``.
        explanation: A single human-readable sentence stating the evidence and
            the rule violated (e.g. ``"3 sub-tasks is within the expected 2-4"``).
    """

    agent: str
    status: Literal["pass", "warning"]
    explanation: str


# ── Individual scoring helpers ───────────────────────────────────────────────

def _score_orchestrator(m: RunMetrics) -> Verdict:
    n = m.orchestrator_subtask_count
    if n < ORCH_MIN_SUBTASKS or n > ORCH_MAX_SUBTASKS:
        return Verdict(
            "orchestrator", "warning",
            f"decomposed the query into {n} sub-task(s); expected "
            f"{ORCH_MIN_SUBTASKS}-{ORCH_MAX_SUBTASKS}.",
        )
    if not m.orchestrator_loop_terminated:
        return Verdict(
            "orchestrator", "warning",
            "the dispatcher loop did not mark every sub-task checked "
            "(risk of getting stuck).",
        )
    return Verdict(
        "orchestrator", "pass",
        f"produced {n} sub-task(s) and the dispatcher loop terminated cleanly.",
    )


def _score_search(m: RunMetrics) -> Verdict:
    # Search's job is purely "return results for every assigned sub-task".
    # Returning nothing for some of them means the downstream notes for those
    # sub-tasks had nothing to work from.
    if m.subtasks_with_empty_results:
        return Verdict(
            "search", "warning",
            f"returned 0 results for {len(m.subtasks_with_empty_results)} sub-task(s): "
            + ", ".join(repr(s) for s in m.subtasks_with_empty_results),
        )
    if m.search_calls == 0 and m.orchestrator_subtask_count > 0:
        return Verdict(
            "search", "warning",
            "made 0 searches despite sub-tasks awaiting processing.",
        )
    return Verdict(
        "search", "pass",
        f"made {m.search_calls} search call(s); every sub-task yielded results.",
    )


def _score_summarizer(m: RunMetrics) -> Verdict:
    missing = m.orchestrator_subtask_count - m.subtasks_summarized
    if missing > 0:
        return Verdict(
            "summarizer", "warning",
            f"produced no summary note for {missing} sub-task(s).",
        )
    if m.trivial_summaries:
        return Verdict(
            "summarizer", "warning",
            "produced empty/trivial summaries for "
            + ", ".join(repr(s) for s in m.trivial_summaries),
        )
    if m.summaries_not_condensed:
        return Verdict(
            "summarizer", "warning",
            "did not condense the raw results for "
            + ", ".join(repr(s) for s in m.summaries_not_condensed),
        )
    return Verdict(
        "summarizer", "pass",
        "condensed every sub-task's raw results into a non-trivial note.",
    )


def _score_fact_checker(m: RunMetrics) -> Verdict:
    # Note: a high flag count is an *informational* warning, not necessarily a
    # failure — the Fact-Checker is doing its job when it flags. The signal we
    # act on is "systematically shaky summaries" (> half of sub-tasks flagged).
    if m.orchestrator_subtask_count > 0 and m.fact_check_flags_ratio > FACT_CHECK_RATIO_THRESHOLD:
        return Verdict(
            "fact_checker", "warning",
            f"raised flags on {m.fact_check_flags_ratio:.0%} of sub-tasks "
            f"({m.fact_check_flags_total} flag(s) total) - upstream summaries "
            "may be systematically weak.",
        )
    return Verdict(
        "fact_checker", "pass",
        f"raised {m.fact_check_flags_total} flag(s); within expected ratio.",
    )


def _score_writer(m: RunMetrics) -> Verdict:
    if m.subtasks_not_covered:
        return Verdict(
            "writer", "warning",
            "final answer never mentions "
            + ", ".join(repr(s) for s in m.subtasks_not_covered),
        )
    if m.answer_length < ANSWER_MIN_LEN or m.answer_length > ANSWER_MAX_LEN:
        return Verdict(
            "writer", "warning",
            f"answer length {m.answer_length} chars is outside the expected "
            f"{ANSWER_MIN_LEN}-{ANSWER_MAX_LEN} range.",
        )
    return Verdict(
        "writer", "pass",
        f"answer covers every sub-task and is {m.answer_length} chars long.",
    )


def _score_overall(m: RunMetrics) -> Verdict:
    if not m.completed_successfully:
        return Verdict(
            "overall", "warning",
            "graph did not complete successfully (errored or never reached "
            "the writer node).",
        )
    if m.total_node_transitions == 0:
        return Verdict(
            "overall", "warning",
            "no node transitions were recorded — nothing appears to have run.",
        )
    return Verdict(
        "overall", "pass",
        f"graph completed successfully in {m.total_run_time_s:.2f}s across "
        f"{m.total_node_transitions} node transition(s).",
    )


# ── Public entry point ───────────────────────────────────────────────────────

def score(metrics: RunMetrics) -> Sequence[Verdict]:
    """
    Produce one :class:`Verdict` per agent (plus an overall verdict) for a run.

    Args:
        metrics: A ``RunMetrics`` from :mod:`evaluation.collector`.

    Returns:
        An ordered list of verdicts: orchestrator, search, summarizer,
        fact_checker, writer, overall.

    Deterministic and side-effect free by construction: same metrics in, same
    verdicts out — which is what you want when reports are committed to a file
    and diffed between runs.
    """
    return [
        _score_orchestrator(metrics),
        _score_search(metrics),
        _score_summarizer(metrics),
        _score_fact_checker(metrics),
        _score_writer(metrics),
        _score_overall(metrics),
    ]