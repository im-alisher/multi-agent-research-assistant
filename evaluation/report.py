"""
evaluation/report.py — Render ``RunMetrics`` + verdicts into a readable report.

Markdown was chosen over plain text for two small reasons: (1) headers and
tables render cleanly in most terminals AND on GitHub/editors, and (2) it lets
the same document serve equally well as the printed console summary and as the
saved file in ``reports/`` — no separate "console" vs "file" formats to keep in
sync. The text itself is plain enough that it reads fine even where markdown
rendering is unavailable.

Two entry points:

* ``generate_report(metrics, verdicts) -> str`` — pure: builds the markdown
  string, never touches disk. Easy to unit-test.
* ``save_report(text) -> Path`` — writes the string into ``reports/`` with a
  unique per-run filename. Pure file I/O, kept separate so tests can call the
  generator without littering the repo with files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from evaluation.metrics import RunMetrics
from evaluation.scorer import Verdict

# Reports are saved one-per-run under the project root's reports/ directory.
# Same pattern as graph/logger.py's LOG_DIR: locate relative to this file so
# it works regardless of which directory the process was launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = _PROJECT_ROOT / "reports"


def _verdict_badge(status: str) -> str:
    """Short status marker for the report table (avoids any emoji/cp issues)."""
    return "PASS" if status == "pass" else "WARN"


def generate_report(
    metrics: RunMetrics,
    verdicts: Sequence[Verdict],
    now: datetime | None = None,
) -> str:
    """
    Build the full evaluation report as a markdown string.

    The report is structured for a human to skim in ~30 seconds:

    1. A one-line overall verdict (did it finish, how long, how many hops).
    2. A per-agent table, one row per verdict, with the PASS/WARN marker.
    3. A 'measurements' section listing the interesting raw facts per agent
       (the sub-task list with per-sub-task result/flag counts) so a warning is
       never an unexplained one-liner.

    Args:
        metrics: The measured run facts (from ``evaluation.collector.collect``).
        verdicts: The per-agent judgements (from ``evaluation.scorer.score``).
        now: Timestamp for the header. Defaults to ``datetime.now()``; passed in
            explicitly so tests can make output deterministic.

    Returns:
        The report text (markdown, UTF-8).
    """
    ts = now or datetime.now()
    when = ts.strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    add = lines.append

    add(f"# Agent Evaluation Report")
    add(f"")
    add(f"- **Query:** {metrics.original_query}")
    add(f"- **Run at:** {when}")
    add(f"- **Completed:** {'yes' if metrics.completed_successfully else 'NO - errored/stuck'}")
    add(f"- **Total run time:** {metrics.total_run_time_s:.2f}s")
    add(f"- **Node transitions:** {metrics.total_node_transitions}")
    add(f"")
    add(f"## Verdicts")
    add(f"")
    add(f"| Agent | Status | Explanation |")
    add(f"| --- | --- | --- |")
    for verdict in verdicts:
        add(f"| {verdict.agent} | {_verdict_badge(verdict.status)} | {verdict.explanation} |")
    add(f"")
    add(f"## Measurements")
    add(f"")

    # Per-sub-task breakdown: the single richest table in the report — it shows,
    # for every sub-task, how many raw results Search brought back, how many
    # Fact-check flags it attracted, and whether the final answer covered it.
    add(f"| Sub-task | Results | Flagged | In answer |")
    add(f"| --- | --- | --- | --- |")
    for st, flags in metrics.flags_by_subtask.items():
        covered = st not in metrics.subtasks_not_covered
        add(f"| {st} | {metrics.results_by_subtask.get(st, 0)} | {flags} | {'yes' if covered else 'no'} |")
    add(f"")

    add(f"**Search:** {metrics.search_calls} call(s); sub-tasks with empty results: "
        f"{', '.join(repr(s) for s in metrics.subtasks_with_empty_results) or 'none'}.")
    add(f"")
    add(f"**Summarizer:** {metrics.subtasks_summarized}/{metrics.orchestrator_subtask_count} "
        f"sub-tasks summarised; not condensed: "
        f"{', '.join(repr(s) for s in metrics.summaries_not_condensed) or 'none'}; trivial: "
        f"{', '.join(repr(s) for s in metrics.trivial_summaries) or 'none'}.")
    add(f"")
    add(f"**Fact-checker:** {metrics.fact_check_flags_total} flag(s) total.")
    add(f"")
    add(f"**Writer:** answer is {metrics.answer_length} chars; sub-tasks not covered: "
        f"{', '.join(repr(s) for s in metrics.subtasks_not_covered) or 'none'}.")
    add(f"")
    add(f"*Rule-based scoring: each verdict is a transparent if/then over the "
        f"measurements above (see evaluation/scorer.py). No learned model is involved.*")
    add(f"")

    return "\n".join(lines)


def save_report(text: str, directory: Path | None = None) -> Path:
    """
    Write ``text`` to ``reports/eval_<timestamp>.md`` and return its path.

    Args:
        text: The report body (e.g. the output of :func:`generate_report`).
        directory: Where to save; defaults to the project's ``reports/`` dir.

    Returns:
        The resulting file path.

    The timestamp is second-resolution; two runs within the same second would
    collide, which is vanishingly unlikely for a manual CLI tool. If you ever
    need sub-second uniqueness, prepend a run id.
    """
    dir_path = Path(directory) if directory else REPORT_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dir_path / f"eval_{stamp}.md"
    dest.write_text(text, encoding="utf-8")
    return dest