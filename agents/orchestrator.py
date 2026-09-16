"""
agents/orchestrator.py — The orchestrator node and its routing helpers.

This module contains:

1. ``SubtaskPlan``          – structured-output schema the LLM must fill in
                               when decomposing the user's query.
2. ``decompose_query``       – calls the LLM to produce 2-4 sub-tasks.
3. ``make_orchestrator_node``– returns a LangGraph node function that (a) runs
                               decomposition the first time it is entered, and
                               (b) on every visit hands the next incomplete
                               sub-task to the rest of the graph by writing it
                               into ``state["current_subtask"]``.
4. ``dispatch``             – the conditional-edge function used by the graph
                               immediately after the orchestrator node to
                               decide which node runs next.

Routing design (the "dispatcher" pattern):
    START  ──> orchestrator ──(conditional: dispatch)──> search
                                                              │
                                                              v
                                                         summarizer
                                                              │
                                                              v
                                                        fact_checker
                                                              │
                                                              v
                                                     (re-enters orchestrator)
                                                              ...
                                                    orchestrator picks next
                                                     or dispatch -> WRITER
                                                     when no tasks remain

The Orchestrator genuinely "delegates" each sub-task: the Search, Summarizer,
and Fact-Checker nodes never see the full query — they only see the sub-task
identified by ``state["current_subtask"]``. The Orchestrator is the only node
that holds the big picture (all sub-tasks + which ones are done).
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from graph.state import ResearchState

# Maximum number of sub-tasks the LLM may return. 4 is a reasonable upper
# bound — beyond that the user experience degrades (too many parallel threads
# make the final answer hard to read).
MAX_SUBTASKS = 4


# ── Structured output schema ─────────────────────────────────────────────────

class SubtaskPlan(BaseModel):
    """
    The shape ``llm.with_structured_output()`` must produce when asked to
    decompose the user's query.

    ``langchain-groq`` + Groq's ``llama-3.3-70b-versatile`` model supports
    JSON mode via function calling.  LangChain calls the model with a schema
    matching this class and expects the model to fill it in; any extra keys the
    model produces are silently ignored.
    """
    subtasks: list[str] = Field(
        description=(
            "A list of 2 to 4 concrete, independent, and researchable "
            "sub-questions that together fully answer the original query."
        )
    )


# ── System prompt ────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a research planning agent. Your job is to take a broad research \
question and decompose it into a small number of concrete, independent, and \
researchable sub-questions that together fully answer the original query.

Rules:
* Return between 2 and {max_subtasks} sub-questions (inclusive).
* Each sub-question must be self-contained — a web search on a single \
sub-question should yield enough information to answer it on its own.
* Cover the breadth of the original question without unnecessary overlap.
* If a sub-question is too vague to search (e.g. "tell me everything"), \
refine it into something specific enough for a web search.
"""


# ── Decomposition function ───────────────────────────────────────────────────

def decompose_query(llm: BaseChatModel, query: str) -> list[str]:
    """
    Ask the LLM to break ``query`` into a list of concrete sub-questions.

    Args:
        llm: The ChatGroq (or compatible) model instance.
        query: The user's original research question.

    Returns:
        A list of 2-4 stripped sub-question strings.

    Raises:
        LLMError: If the Groq API is unreachable or the key is invalid.
        OutputParserException: If the model fails to produce valid JSON
            matching the ``SubtaskPlan`` schema (extremely unlikely with a
            well-behaved prompt, but possible under rate limits).
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("human", "{query}"),
    ])
    chain = prompt | llm.with_structured_output(SubtaskPlan)
    plan = chain.invoke({
        "query": query,
        "max_subtasks": MAX_SUBTASKS,
    })
    # Clamp and clean the output.  The model may occasionally return 1 or 5+.
    return [s.strip() for s in plan.subtasks if s.strip()][:MAX_SUBTASKS]


# ── Helpers used by the node and the conditional edge ────────────────────────

def _is_checked(subtask: str, checked: list[str]) -> bool:
    """Return ``True`` when a sub-task has completed the full pipeline."""
    return subtask in checked


def _next_pending_subtask(state: ResearchState) -> str | None:
    """
    Walk the ordered sub-task list and return the first one whose pipeline is
    not yet complete (i.e. not in ``checked_subtasks``).

    Returns ``None`` when every sub-task has been processed — the signal to
    the dispatch conditional edge that it is time to route to the Writer.
    """
    checked = state.get("checked_subtasks") or []
    for subtask in state.get("subtasks", []):
        if not _is_checked(subtask, checked):
            return subtask
    return None


# ── The node function ────────────────────────────────────────────────────────

def make_orchestrator_node(
    llm: BaseChatModel,
) -> Callable[[ResearchState], dict[str, Any]]:
    """
    Return a LangGraph node that acts as the orchestrator / dispatcher.

    The node is *idempotent*: if called when sub-tasks already exist in state
    (e.g. during a test that pre-fills the state), it skips decomposition and
    goes straight to handing out the next sub-task.

    The ``current_subtask`` field written by this node is read by the Search,
    Summarizer, and Fact-Checker nodes to know which sub-task to operate on.
    """
    def orchestrator_node(state: ResearchState) -> dict[str, Any]:
        updates: dict[str, Any] = {"current_step": "orchestrator"}

        # First visit: decompose the original query.
        if not state.get("subtasks"):
            subtasks = decompose_query(llm, state["original_query"])
            updates["subtasks"] = subtasks
            # current_subtask depends on the freshly-written subtasks; since
            # the updates dict is applied to state *after* this function
            # returns, we recompute it explicitly here.
            updates["current_subtask"] = subtasks[0] if subtasks else ""
            return updates

        # Subsequent visits (after fact_checker → back to orchestrator):
        # just hand out the next pending sub-task, or empty string when done.
        pending = _next_pending_subtask(state)
        updates["current_subtask"] = pending or ""
        return updates

    return orchestrator_node


# ── Conditional-edge router ──────────────────────────────────────────────────

NODE_SEARCH = "search"
NODE_SUMMARIZER = "summarizer"
NODE_FACT_CHECKER = "fact_checker"
NODE_WRITER = "writer"


def dispatch(state: ResearchState) -> str:
    """
    Decide which node to route to immediately after the Orchestrator runs.

    If the Orchestrator wrote a non-empty ``current_subtask`` into the state,
    it means there is still work to do and we route to the Search node.

    If ``current_subtask`` is empty (the Orchestrator found no remaining
    pending sub-tasks), every sub-task has been fully processed and we route
    straight to the Writer to assemble the final answer.

    Note: this function is a pure read-only function of the state — it cannot
    itself write to state. It returns *only* the name of the next node. The
    conditional-edge machinery of LangGraph takes care of invoking that node.
    """
    if state.get("current_subtask"):
        return NODE_SEARCH
    return NODE_WRITER