"""
agents/fact_checker_agent.py — The Fact-Checker node.

After a sub-task has been searched and summarised, this node verifies the
summary *against the raw evidence it came from*. Its job is to catch claims in
the note that are not actually supported by (or that contradict) the source
snippets the Search node gathered.

This is a classic "adversarial reader" pattern: the Summarizer is encouraged to
be concise and could subtly over-condense; the Fact-Checker applies an
explicitly sceptical lens. It returns:

* ``flags``  – human-readable warnings (unsupported / exaggerated /
               contradictory claim) appended to ``state["fact_check_flags"]``.
* ``check_id``/``report`` – a short plain-English review for the logs.

It also appends the sub-task to ``state["checked_subtasks"]``. That is the
completion marker the Orchestrator's router uses to decide the sub-task is done
and to stop re-dispatching it.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from graph.state import ResearchState

MAX_RESULTS_IN_PROMPT = 5


class FactCheckReport(BaseModel):
    """
    Structured output the Fact-Checker LLM must produce.

    ``flags``   -> list of warning strings, one per unsupported/exaggerated/
                   contradictory finding. Empty list means "nothing to flag".
    ``report``  -> one or two plain-English sentences summarising the verdict.
    """
    flags: list[str] = Field(
        description=(
            "Warnings for claims in the summary that are unsupported by, "
            "exaggerated relative to, or in contradiction with the search "
            "results. Empty list if the summary is fully supported."
        )
    )
    report: str = Field(
        description="A short plain-language summary of the fact-check verdict."
    )


FACT_CHECKER_SYSTEM_PROMPT = """\
You are a rigorous fact-checker for a research pipeline. You receive a \
research sub-question, a summary note claiming to answer it, and the raw web \
search results that the note claims to be based on.

Compare the note claim-by-claim against the provided search results and:
* FLAG any claim that has no supporting source among the provided results.
* FLAG any claim that overstates what the sources actually say.
* FLAG any internal contradiction within the note or between the note and a \
source.
* Do NOT flag on the basis of outside knowledge — a claim can only be checked \
against the sources you are given. (A note may still be factually wrong about \
the real world and pass this check; that is expected and out of scope here.)

Output exactly one flag per problem found. If everything is supported, leave \
the flags list empty.
"""


def make_fact_checker_node(
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    """
    Return a LangGraph node that fact-checks the current sub-task's summary.

    Args:
        llm: The chat model used for the review.

    Returns:
        A node function producing a partial state update with ``fact_check_flags``
        (accumulated) and ``checked_subtasks`` (accumulated).
    """
    def fact_checker_node(state: ResearchState) -> dict[str, Any]:
        subtask = state.get("current_subtask", "")
        note = state.get("summarized_notes", {}).get(subtask, "")
        results = state.get("raw_search_results", {}).get(subtask, [])

        if not subtask:
            print("[fact_checker_agent] no current_subtask in state; skipping")
            return {"current_step": "fact_checker"}

        snippet_lines = [
            f"- [{r.get('title', '')}]({r.get('url', '')})\n  {r.get('snippet', '')}"
            for r in results[:MAX_RESULTS_IN_PROMPT]
        ]
        results_text = "\n".join(snippet_lines) or "(no sources provided)"

        prompt = ChatPromptTemplate.from_messages([
            ("system", FACT_CHECKER_SYSTEM_PROMPT),
            (
                "human",
                "Sub-question: {subtask}\n\n"
                "Summary note being checked:\n{note}\n\n"
                "Raw search results the note claims to be based on:\n{results}",
            ),
        ])
        messages = prompt.format_messages(
            subtask=subtask,
            note=note,
            results=results_text,
        )

        structured = llm.with_structured_output(FactCheckReport)
        report = structured.invoke(messages)

        return {
            "current_step": "fact_checker",
            # ``operator.add`` reducers on both fields will *append* these to
            # whatever flags/checked sub-tasks accumulated so far.
            "fact_check_flags": list(report.flags),
            "checked_subtasks": [subtask],
        }

    return fact_checker_node