"""
main.py — Command-line entry point for the Multi-Agent Research Assistant.

Usage:
    python main.py "Will AI agents replace software engineers?"
    python main.py                 (prompts interactively for a question)
    python main.py --no-eval ...   (skips the post-run evaluation pass)

What happens:
    1. Build the real graph (Groq LLM + DuckDuckGo search).
    2. Run the user's question through the full pipeline.
    3. Print a short run summary, then the final answer.
    4. Save a per-run copy of the agent communication log to ``logs/``.
    5. Unless ``--no-eval`` is given, run the agent evaluation framework
       (collector + scorer + report, see ``evaluation/``) and print/save the
       per-agent pass/warning report to ``reports/``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import GroqError

from evaluation.collector import capture_agent_log, collect, entries_from_records
from evaluation.report import generate_report, save_report
from evaluation.scorer import score
from graph.build_graph import build_default_graph
from graph.logger import LOG_DIR, LOG_FILE, get_logger

# All fields are optional in the schema, but being explicit up front makes the
# first graph step (the Orchestrator's decomposition) clearer to follow.
INITIAL_STATE = {
    "original_query": "",        # replaced below with the user's question
    "subtasks": [],
    "raw_search_results": {},
    "summarized_notes": {},
    "fact_check_flags": [],
    "checked_subtasks": [],
    "current_subtask": "",
    "final_answer": "",
    "current_step": "",
}


def save_run_log() -> Path | None:
    """
    Copy ``logs/agent_comm.log`` into a timestamped per-run file.

    The live communication logger appends to one shared file; this gives each
    CLI invocation its own snapshotted copy (e.g. ``run_20260916_120000.log``)
    so a later run never muddies an earlier one.

    Returns:
        The path of the saved copy, or ``None`` if there was nothing to save.
    """
    src = Path(LOG_FILE)
    if not src.exists():
        return None
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(LOG_DIR) / f"run_{stamp}.log"
    shutil.copyfile(src, dest)
    return dest


def evaluate_run(
    final_state: dict,
    records: list[logging.LogRecord],
) -> str:
    """
    Run the full evaluation pipeline on a completed (or attempted) run.

    Pipeline: raw log records -> structured entries (collector), entries +
    final state -> ``RunMetrics`` (collector), metrics -> verdicts (scorer),
    both -> markdown report (report). The report is printed to stdout and
    saved to ``reports/eval_<timestamp>.md``.

    Args:
        final_state: The graph's final state. On an errored run this may be
            the pre-run state; the collector marks ``completed_successfully``
            as False and the report shows an 'errored' overall verdict.
        records: The run's captured ``logging.LogRecord`` objects (from the
            ``LogRecordCollector`` attached during ``graph.invoke``).

    Returns:
        The generated report text (also printed and saved).
    """
    entries = entries_from_records(records)
    metrics = collect(final_state, entries)
    verdicts = score(metrics)
    report_text = generate_report(metrics, verdicts)

    print("\n" + "=" * 70)
    print("AGENT EVALUATION SUMMARY")
    print("=" * 70)
    print(report_text)
    saved = save_report(report_text)
    print(f"Full evaluation report saved to: {saved}")
    return report_text


def run_research(state: dict) -> dict:
    """
    Invoke the compiled research graph on a pre-filled state dict.

    Args:
        state: A state dict with at least ``original_query`` set (usually
            ``INITIAL_STATE`` copied and augmented by ``main``).

    Returns:
        The final state dict (Populated: subtasks, notes, flags, final_answer).
    """
    graph = build_default_graph()
    return graph.invoke(state)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and execute a research run."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant: ask a question and let "
                    "five cooperating agents research it.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="The research question. If omitted, you are prompted interactively.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the post-run agent evaluation pass for a faster run.",
    )
    args = parser.parse_args(argv)

    load_dotenv()  # ensure GROQ_API_KEY is present before building the graph

    question = args.question
    if not question:
        question = input("Your research question: ").strip()
    if not question:
        print("No question provided. Nothing to do.")
        return 1

    print(f"\nResearching: {question}\n")

    # Build the state up front so that, if the run fails, we still have an
    # (empty) state object to feed the evaluation collector — it reports the
    # failed run as "errored" instead of crashing the CLI.
    state = dict(INITIAL_STATE)
    state["original_query"] = question

    error: Exception | None = None
    try:
        # The LogRecordCollector is attached around invoke so we capture ONLY
        # this run's communication entries (the shared file logger accumulates
        # every run). On an exception the `with` block unwinds but `collector`
        # stays bound, so its records are available to evaluate the failure too.
        with capture_agent_log(get_logger()) as collector:
            final_state = run_research(state)
    except (GroqError, ValueError) as exc:
        # GroqError: missing/invalid GROQ_API_KEY or an upstream API problem.
        error = exc
        final_state = state  # nothing completed; evaluate this as an errored run.

    if error is not None:
        print(f"ERROR: {error}")
        print("Is GROQ_API_KEY set in your environment or .env file?")

    # ── Report the outcome ────────────────────────────────────────────────
    if error is None:
        subtasks = final_state.get("subtasks", [])
        print("=" * 70)
        print(f"Decomposed into {len(subtasks)} sub-tasks:")
        for i, st in enumerate(subtasks, 1):
            print(f"  {i}. {st}")
        if final_state.get("fact_check_flags"):
            print(f"\n{len(final_state['fact_check_flags'])} fact-check warning(s) "
                  "were attached to the answer below as caveats.")

        print("=" * 70)
        print(final_state.get("final_answer", "(no answer was produced)"))

    # ── Persist this run's communication trail ────────────────────────────
    saved = save_run_log()
    if saved:
        print(f"\nAgent communication log saved to: {saved}")

    # ── Evaluate the run (collector + scorer + report) ────────────────────
    if not args.no_eval:
        evaluate_run(final_state, collector.records)

    return 1 if error is not None else 0


if __name__ == "__main__":
    sys.exit(main())