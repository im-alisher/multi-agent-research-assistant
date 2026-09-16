"""
agents/summarizer_agent.py — The Summarizer node.

The Search node hands this node a pile of raw search results for the current
sub-task (``state["raw_search_results"][current_subtask]``). The Summarizer's
job is to condense that pile into a short, factual, self-contained note that
the Fact-Checker and the Writer can consume without re-reading every snippet.

Design notes:

* It is *unfaithful-by-prompt*: the system prompt explicitly forbids
  inventing facts, interpreting sources, or adding background knowledge. The
  whole point is that it can only restate evidence that actually appeared in
  the search results.
* Just like the other agents, this node is a factory that captures its LLM at
  construction time, so tests can inject a fake model.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from graph.state import ResearchState

# Cap the number of results we stuff into the prompt — enough to summarise
# properly, few enough to fit comfortably in the model's context window.
MAX_RESULTS_IN_PROMPT = 5


SUMMARIZER_SYSTEM_PROMPT = """\
You are a research summariser. You are given the raw web search results for a \
single research sub-question. Produce a concise, factual summary note.

Rules:
* Only use facts that actually appear in the provided results. Do NOT add \
background knowledge, speculation, or interpretation.
* Cover the most important / repeated points across the results, in compact \
bullet-point style (2-6 bullets).
* End the note with a short "Sources:" line listing the URLs you used.
* If the results are empty or contain nothing relevant, say \
"No useful information was found for this sub-task from the provided sources."
"""


def make_summarizer_node(
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    """
    Return a LangGraph node that summarises raw search results into a note.

    Args:
        llm: The chat model used to write the summary note.

    Returns:
        A node function ``(state) -> {"current_step": ..., "summarized_notes": {...}}``.
    """
    def summarizer_node(state: ResearchState) -> dict[str, Any]:
        subtask = state.get("current_subtask", "")
        results = state.get("raw_search_results", {}).get(subtask, [])

        if not subtask:
            print("[summarizer_agent] no current_subtask in state; skipping")
            return {"current_step": "summarizer"}

        # Build the prompt from a trimmed view of the raw results.
        snippet_lines = [
            f"- [{r.get('title', '')}]({r.get('url', '')})\n  {r.get('snippet', '')}"
            for r in results[:MAX_RESULTS_IN_PROMPT]
        ]
        results_text = "\n".join(snippet_lines) or "(no results returned)"

        prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARIZER_SYSTEM_PROMPT),
            ("human", "Sub-question: {subtask}\n\nSearch results:\n{results}"),
        ])
        messages = prompt.format_messages(
            subtask=subtask,
            results=results_text,
        )

        answer = llm.invoke(messages)
        note = answer.content if hasattr(answer, "content") else str(answer)

        return {
            "current_step": "summarizer",
            "summarized_notes": {subtask: note.strip()},
        }

    return summarizer_node