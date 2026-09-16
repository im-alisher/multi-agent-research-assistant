"""
tools/search_tool.py — A thin, mockable wrapper around the DuckDuckGo web search.

The intent of this module is to decouple the Search agent from any particular
search provider. The agent only knows about the function ``web_search`` and the
``SearchResult`` shape it returns; if we later swap DuckDuckGo for Tavily or
Bing, we only change this one file — nothing in ``agents/`` or ``graph/`` cares.

We wrap the raw ``ddgs`` call in a plain Python function (rather than a LangChain
``Tool``) on purpose:

* It is trivial to mock in tests — no LiveAPI connection, no backend config.
* It keeps the agent code simple and readable, which is the goal of this project.

Every parameter is type-hinted and the return type is ``list[SearchResult]`` so
the contract with ``graph.state`` is enforced statically.
"""

from __future__ import annotations

from typing import Callable

from ddgs import DDGS

from graph.state import SearchResult

# Number of results to ask DuckDuckGo for. 5 is enough for summarisation
# without drowning the LLM prompt in noise.
MAX_RESULTS = 5


def _format_results(raw: list[dict]) -> list[SearchResult]:
    """
    Normalise the provider's raw dicts into our stable ``SearchResult`` shape.

    The ``ddgs`` library returns a list of dicts that vary slightly between
    versions (e.g. they may add or rename keys). By mapping explicitly here we
    (a) guarantee the exact keys the rest of the project expects, and
    (b) drop fields (e.g. ``body``, ``date``) the agents never use.
    """
    formatted: list[SearchResult] = []
    for item in raw:
        formatted.append(
            SearchResult(
                title=str(item.get("title", "")).strip(),
                snippet=str(item.get("body") or item.get("snippet") or "").strip(),
                url=str(item.get("href") or item.get("url") or "").strip(),
            )
        )
    return formatted


def web_search(query: str, max_results: int = MAX_RESULTS) -> list[SearchResult]:
    """
    Run a web search on ``query`` via DuckDuckGo and return normalised results.

    Args:
        query: The search string (normally one of the Orchestrator's sub-tasks).
        max_results: Maximum number of results to return.

    Returns:
        A list of ``SearchResult`` dicts ``{"title", "snippet", "url"}``.
        Empty list on failure — the Search agent treats "no results" as a valid
        outcome (the summariser will simply note that nothing was found).

    Raises:
        Nothing. Network / provider errors are swallowed and logged so a single
        flaky search never crashes the whole graph. (This is a deliberate
        fail-open design choice; see the README's "What I would improve".)
    """
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return _format_results(raw)
    except Exception as exc:  # noqa: BLE001 - provider errors are expected
        print(f"[search_tool] search failed for {query!r}: {exc}")
        return []


# Type alias for the search function, so callers can reference web_search by its
# full name and also pass it around as a value (e.g. to inject a fake in tests).
SearchToolCallable = Callable[[str, int], list[SearchResult]]