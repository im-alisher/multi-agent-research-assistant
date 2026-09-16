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

# The model identifier on Groq's free tier. "versatile" is Groq's label for a
# model that handles both chat and tool/function calling well.
#
# llama-3.3-70b is a strong, cheap open model; if a better free-tier model
# becomes available, this is the only place that needs changing.
MODEL_NAME = "llama-3.3-70b-versatile"

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