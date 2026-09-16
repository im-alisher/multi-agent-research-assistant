"""
agents/writer_agent.py — The Writer node.

The last stop. Every sub-task has been searched, summarised, and fact-checked.
The Writer receives:

* all of the verified summary notes (``state["summarized_notes"]``),
* the fact-check warnings raised along the way (``state["fact_check_flags"]``),

and composes the single, structured, readable answer the user actually sees.

The fact-check flags are not discarded: the Writer is told to treat any
flagged claim with an explicit caveat (e.g. "this claim is not supported by
the available sources"), so the user gets an honest answer instead of a
confident-sounding one. Flagging + caveating is a much cheaper stand-in for
full retrieval-augmented verification, which is appropriate for this project.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from graph.state import ResearchState


WRITER_SYSTEM_PROMPT = """\
You are the final writer for a research pipeline. Summarised, fact-checked \
notes for each sub-question and a list of fact-check warnings are provided.

Write a clear, well-structured final answer to the ORIGINAL question.

Rules:
* Organise the answer around the sub-questions: an intro, one section per \
sub-task, then a one-paragraph conclusion.
* Base every claim directly on the provided notes. Do not invent facts or \
sources that are not in the notes.
* If the fact-check warnings list any issue, explicitly caveat the affected \
claim with a short parenthetical such as "(not fully supported by the \
retrieved sources)". If there are no warnings, do not mention fact-checking at \
all.
* Append a "Sources" section listing the URLs referenced in the notes.
* Keep it concise — aim for well under a page of text.
"""


def make_writer_node(
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    """
    Return a LangGraph node that writes the final answer from the notes.

    Args:
        llm: The chat model used to compose the answer.

    Returns:
        A node function producing the ``final_answer`` field plus a
        ``current_step`` marker.
    """
    def writer_node(state: ResearchState) -> dict[str, Any]:
        original_query = state.get("original_query", "")
        notes = state.get("summarized_notes", {})
        flags = state.get("fact_check_flags", [])

        # Present the notes as an ordered, readable transcript of the sub-task.
        notes_text = "\n\n".join(
            f"## Sub-question: {subtask}\n{note}"
            for subtask, note in notes.items()
            if note
        ) or "(no notes were produced — every search returned nothing usable)"

        flags_text = "\n".join(f"- {flag}" for flag in flags) or "(no warnings)"

        prompt = ChatPromptTemplate.from_messages([
            ("system", WRITER_SYSTEM_PROMPT),
            (
                "human",
                "Original question: {query}\n\n"
                "Fact-check warnings:\n{flags}\n\n"
                "Verified notes by sub-question:\n\n{notes}",
            ),
        ])
        messages = prompt.format_messages(
            query=original_query,
            flags=flags_text,
            notes=notes_text,
        )

        answer = llm.invoke(messages)
        final_answer = answer.content if hasattr(answer, "content") else str(answer)

        return {
            "current_step": "writer",
            "final_answer": final_answer.strip(),
        }

    return writer_node