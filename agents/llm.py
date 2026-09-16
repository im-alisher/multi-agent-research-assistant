"""
agents/llm.py — A single shared factory for the LLM used by every agent.

All five agents in this project do their reasoning through the same underlying
model. Keeping the construction in one place means:

* we configure the model exactly once (model name, temperature, key loading);
* swapping the model later (e.g. to GPT or a local model for testing) changes
  one line instead of one line per agent file.

The model is loaded from the ``GROQ_API_KEY`` environment variable via
python-dotenv, so the key never lives in the source tree.
"""

from __future__ import annotations

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# The model identifier on Groq's free tier.
#
# NOTE on selection (checked Sept 2026): Groq's previous
# ``llama-3.3-70b-versatile`` no longer exists on the account this project uses.
# The larger ``openai/gpt-oss-120b`` / ``openai/gpt-oss-20b`` models were
# tested but fail Groq's enforced tool-calling for ``with_structured_output``
# ("Tool choice is required, but model did not call a tool"), which the
# Orchestrator and Fact-Checker depend on.
#
# ``qwen/qwen3.8-27b`` reliably supports both plain chat and tool-calling
# structured output, so it is the current default. If a better free-tier model
# that supports function calling becomes available, change it here — this is
# the only place that needs editing.
MODEL_NAME = "qwen/qwen3.8-27b"

# Low temperature keeps decomposition/summarisation outputs factual-ish and
# reproducible; the value is intentionally modest (not 0.0) so responses are
# not robotically identical every run.
TEMPERATURE = 0.2


def get_llm() -> ChatGroq:
    """
    Build and return a configured ``ChatGroq`` chat model ready to be invoked.

    ``load_dotenv()`` reads ``.env`` from the current working directory (or the
    ``GROQ_API_KEY`` environment variable if already set) and makes the key
    available to the Groq SDK.

    Returns:
        A ``ChatGroq`` instance using ``llama-3.3-70b-versatile``.
    """
    load_dotenv()
    return ChatGroq(model=MODEL_NAME, temperature=TEMPERATURE)