"""
graph/logger.py — Inter-agent communication logger.

The point of this logger is pedagogical: it makes the "message passing"
between agents visible. Because agents in a LangGraph never call each other
directly, the only observable facts about the flow are the *state transitions*
the graph performs between nodes. This module records exactly those.

How it works:
    ``build_graph`` (see ``graph/build_graph.py``) wraps every node function
    with ``wrap_for_logging``. Each time the graph enters a node, the wrapper
    writes a line to ``logs/agent_comm.log`` recording:

        * which agent ran,
        * what the state contained at entry (a compact summary),
        * what partial state updates the node returned (its "message" to the
          rest of the graph).

Because the wrapper only logs and always returns whatever the wrapped node
returned, it is safe to apply/unapply without changing graph behaviour.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

# The logs/ directory lives at the project root (one level up from graph/).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "agent_comm.log")

# Module-level singleton so all agents share one file/stream handler.
_LOGGER: logging.Logger | None = None


def _summarize(value: Any, max_len: int = 200) -> str:
    """
    Turn an arbitrary state value into a short, human-readable string.

    Used so log lines stay readable even when the value is a big list of
    search results or the final answer. Long strings are truncated with ``…``.
    """
    if isinstance(value, dict):
        if not value:
            return "{}"
        parts = [f"{key}={_summarize(val, max_len // 4)}" for key, val in value.items()]
        return "{" + ", ".join(parts[:4]) + ("…" if len(parts) > 4 else "") + "}"
    if isinstance(value, list):
        return f"[{len(value)} item(s)]"
    text = str(value)
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def get_logger() -> logging.Logger:
    """
    Return a configured ``logging.Logger`` writing to ``logs/agent_comm.log``.

    Lazily creates the ``logs/`` directory and attaches exactly one file
    handler; repeated calls return the same logger (no duplicate handlers).
    """
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("agent_comm")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    _LOGGER = logger
    return logger


def wrap_for_logging(
    agent_name: str,
    node: Callable[[Any], dict],
) -> Callable[[Any], dict]:
    """
    Wrap a graph node so its entry/exit is logged to ``logs/agent_comm.log``.

    Args:
        agent_name: Human-readable node name, e.g. "orchestrator".
        node: The original node function ``(state: ResearchState) -> updates``.

    Returns:
        A node function with identical behaviour plus logging.

    The wrapper records:
        * ``>> agent <name> received`` — a summary of the state at entry,
        * ``   agent <name> produced`` — the partial state update returned.

    The two lines bracket every transition in the run, which is exactly the
    communication trail the rest of the project reasons about.
    """
    logger = get_logger()

    def wrapped(state: dict) -> dict:
        logger.info(">> agent: %s | current_step=%s | received: %s",
                    agent_name,
                    _summarize(state.get("current_step")),
                    _summarize(state))
        updates = node(state)
        logger.info("   agent: %s | produced: %s", agent_name, _summarize(updates))
        return updates

    return wrapped