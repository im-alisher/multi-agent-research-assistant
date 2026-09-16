"""
tests/fakes.py — Fake LLM and fake search callable for the test suite.

None of the tests may touch the live Groq API or the live DuckDuckGo service,
so we inject these fakes through the dependency-injection seams the agents
already expose (``make_*_node(llm, ...)`` and ``build_graph(llm, search_fn)``).

``FakeLLM`` mimics just enough of ``langchain``'s chat-model interface to
satisfy the four agents that use an LLM. It determines *which* agent is calling
it by looking for the distinctive system-prompt phrase, then returns canned
data for that agent. ``FakeMessage`` stands in for a LangChain ``AIMessage`` —
the writer/summarizer code only reads ``.content``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.runnables import RunnableLambda

from agents.orchestrator import SubtaskPlan
from agents.fact_checker_agent import FactCheckReport


class FakeMessage:
    """Cheap stand-in for ``AIMessage``: only ``.content`` is ever read."""

    def __init__(self, content: str) -> None:
        self.content = content


def _text_of(prompt_value: object) -> str:
    """Join all message contents (works for prompt values or message lists)."""
    msgs = (
        prompt_value.messages
        if hasattr(prompt_value, "messages")
        else prompt_value
    )
    return " ".join(
        m.content for m in msgs if hasattr(m, "content")
    )


@dataclass
class FakeLLM:
    """Deterministic fake chat model keyed off the system-prompt phrase."""

    subtasks: list[str] = field(default_factory=lambda: [
        "What is LangGraph?",
        "Who created LangGraph?",
    ])
    flags: list[str] = field(default_factory=list)
    note_text: str = "Concise factual note about the sub-task. Sources: http://example.org"
    answer_text: str = "Final answer to the original question."

    def _respond(self, prompt_value: object):
        text = _text_of(prompt_value)
        if "You are a research planning agent" in text:
            return SubtaskPlan(subtasks=self.subtasks)
        if "You are a rigorous fact-checker" in text:
            return FactCheckReport(flags=list(self.flags), report="Checked notes.")
        if "You are the final writer" in text:
            return FakeMessage(self.answer_text)
        # Default branch: the summariser prompt.
        return FakeMessage(self.note_text)

    def with_structured_output(self, schema):
        return RunnableLambda(self._respond)

    def invoke(self, prompt_value):
        return self._respond(prompt_value)


def fake_search(query: str, max_results: int = 5) -> list[dict]:
    """Return one deterministic fake result for any query."""
    return [{
        "title": f"Result about: {query}",
        "snippet": "A synthetic snippet for testing purposes.",
        "url": f"https://example.com/search?q={query}",
    }]