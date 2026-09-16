"""
main.py — Command-line entry point for the Multi-Agent Research Assistant.

Usage:
    python main.py "Will AI agents replace software engineers?"
    python main.py                 (prompts interactively for a question)

What happens:
    1. Build the real graph (Groq LLM + DuckDuckGo search).
    2. Run the user's question through the full pipeline.
    3. Print a short run summary, then the final answer.
    4. Save a per-run copy of the agent communication log to ``logs/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from groq import GroqError

from graph.build_graph import build_default_graph
from graph.logger import LOG_DIR, LOG_FILE

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


def run_research(query: str) -> dict:
    """
    Run ``query`` through the compiled research graph end to end.

    Args:
        query: The user's research question.

    Returns:
        The final state dict (Populated: subtasks, notes, flags, final_answer).
    """
    graph = build_default_graph()
    state = dict(INITIAL_STATE)
    state["original_query"] = query
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
    args = parser.parse_args(argv)

    load_dotenv()  # ensure GROQ_API_KEY is present before building the graph

    question = args.question
    if not question:
        question = input("Your research question: ").strip()
    if not question:
        print("No question provided. Nothing to do.")
        return 1

    print(f"\nResearching: {question}\n")
    try:
        final_state = run_research(question)
    except (GroqError, ValueError) as exc:
        # GroqError: missing/invalid GROQ_API_KEY or an upstream API problem.
        print(f"ERROR: {exc}")
        print("Is GROQ_API_KEY set in your environment or .env file?")
        return 1

    # ── Report the outcome ────────────────────────────────────────────────
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
    return 0


if __name__ == "__main__":
    sys.exit(main())