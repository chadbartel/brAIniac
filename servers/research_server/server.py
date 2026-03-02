"""servers/research_server/server.py

FastMCP research server for brAIniac — Phase 2.

Implements the **IterDRAG** (Iterative Dense Retrieval-Augmented Generation)
research loop:

  1. Query SearXNG for top-N results.
  2. Use Ollama to refine the query into a targeted sub-query.
  3. Re-run SearXNG with the refined query; collect results.
  4. Repeat up to ``RESEARCH_MAX_ITERATIONS`` (default 3).
  5. Synthesise all collected chunks into one cited answer via Ollama.

All LLM calls go directly through ``ollama.Client`` so this server stays
isolated from ``core/`` and does not import any cross-server logic.

Environment variables consumed:
  SEARXNG_HOST              — default: http://localhost:8080
  RESEARCH_MAX_ITERATIONS   — default: 3
  RESEARCH_TOP_N            — default: 5
  OLLAMA_HOST               — default: http://localhost:11434
  OLLAMA_MODEL              — default: llama3.1:8b-instruct-q4_K_M
"""

from __future__ import annotations

# Standard Library
import os
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# Third-Party Libraries
from ollama import Client
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP server init
# ---------------------------------------------------------------------------

mcp: FastMCP = FastMCP(
    "brAIniac-research-server",
    instructions=(
        "Provides iterative deep-research capabilities for brAIniac. "
        "Queries SearXNG, refines sub-queries via Ollama, and returns a "
        "synthesised answer with inline citation URLs."
    ),
)

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_SEARXNG_HOST_DEFAULT = "http://localhost:8080"
_OLLAMA_HOST_DEFAULT = "http://localhost:11434"
_OLLAMA_MODEL_DEFAULT = "llama3.1:8b-instruct-q4_K_M"


def _get_env() -> dict[str, Any]:
    """Read all research-server env vars in one place."""
    return {
        "searxng_host": os.getenv("SEARXNG_HOST", _SEARXNG_HOST_DEFAULT).rstrip("/"),
        "ollama_host": os.getenv("OLLAMA_HOST", _OLLAMA_HOST_DEFAULT),
        "ollama_model": os.getenv("OLLAMA_MODEL", _OLLAMA_MODEL_DEFAULT),
        "max_iterations": int(os.getenv("RESEARCH_MAX_ITERATIONS", "3")),
        "top_n": int(os.getenv("RESEARCH_TOP_N", "5")),
    }


# ---------------------------------------------------------------------------
# SearXNG helper
# ---------------------------------------------------------------------------


def _search_searxng(query: str, top_n: int, searxng_host: str) -> list[dict[str, str]]:
    """Query a local SearXNG instance and return structured results.

    Args:
        query: Search terms.
        top_n: Maximum number of results to return.
        searxng_host: Base URL of the SearXNG service.

    Returns:
        List of result dicts with keys ``url``, ``title``, ``snippet``.

    Raises:
        urllib.error.URLError: If the SearXNG service is unreachable.
        ValueError: If the response JSON is malformed.
    """
    params = urllib.parse.urlencode(
        {"q": query, "format": "json", "categories": "general"}
    )
    url = f"{searxng_host}/search?{params}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            if resp.status != 200:
                raise ValueError(f"SearXNG returned HTTP {resp.status}")
            data: dict[str, Any] = json.loads(resp.read())
    except urllib.error.URLError:
        raise
    except Exception as exc:
        raise ValueError(f"SearXNG response error: {exc}") from exc

    if "error" in data:
        raise ValueError(f"SearXNG error response: {data['error']}")

    results: list[dict[str, str]] = []
    for item in data.get("results", [])[:top_n]:
        results.append(
            {
                "url": str(item.get("url", "")),
                "title": str(item.get("title", "")),
                "snippet": str(item.get("content", item.get("snippet", ""))),
            }
        )
    return results


# ---------------------------------------------------------------------------
# IterDRAG implementation
# ---------------------------------------------------------------------------


def _format_results_block(results: list[dict[str, str]], iteration: int) -> str:
    """Format a list of SearXNG results into a context block for the LLM.

    Args:
        results: Structured search results.
        iteration: The current iteration number (1-based) for labelling.

    Returns:
        Plain-text block suitable for inclusion in an Ollama chat message.
    """
    lines: list[str] = [f"=== Search Results (iteration {iteration}) ==="]
    for idx, r in enumerate(results, 1):
        lines.append(
            f"[{idx}] {r['title']}\n" f"    URL: {r['url']}\n" f"    {r['snippet']}"
        )
    return "\n".join(lines)


def _refine_query(
    original_query: str,
    context_block: str,
    ollama_client: Client,
    model: str,
) -> str:
    """Ask Ollama to generate a refined sub-query based on gathered evidence.

    Args:
        original_query: The user's original research question.
        context_block: The formatted results gathered so far.
        ollama_client: An initialised ``ollama.Client`` instance.
        model: Ollama model identifier.

    Returns:
        A more focused query string (or the original if the LLM's response
        cannot be parsed cleanly).
    """
    prompt = (
        f"You are a research assistant. The user wants to know: {original_query!r}\n\n"
        f"Based on this evidence:\n{context_block}\n\n"
        "Generate ONE concise search query (max 10 words) that would retrieve "
        "the most important missing information. Return only the query, no explanation."
    )
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        refined: str = (
            getattr(resp.get("message", {}), "content", None)
            or (resp.get("message", {}) or {}).get("content", "")
            or original_query
        ).strip()
        return refined if refined else original_query
    except Exception as exc:
        logger.warning("Query refinement failed: %s — using original query.", exc)
        return original_query


def _synthesise(
    original_query: str,
    all_results: list[dict[str, str]],
    ollama_client: Client,
    model: str,
) -> str:
    """Synthesise all gathered results into one coherent cited answer.

    Args:
        original_query: The user's original research question.
        all_results: Deduplicated list of all results from all iterations.
        ollama_client: An initialised ``ollama.Client`` instance.
        model: Ollama model identifier.

    Returns:
        A human-readable answer with inline citation URLs.
    """
    sources_block = "\n".join(
        f"[{i+1}] {r['title']} — {r['url']}\n    {r['snippet']}"
        for i, r in enumerate(all_results)
    )
    prompt = (
        f"You are a research analyst. Answer this question: {original_query!r}\n\n"
        f"Use ONLY the sources below. Cite sources inline using their [N] number.\n\n"
        f"{sources_block}\n\n"
        "Write a clear, factual answer in 3-6 sentences with inline citations."
    )
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (
            getattr(resp.get("message", {}), "content", None)
            or (resp.get("message", {}) or {}).get("content", "")
            or "Unable to synthesise an answer from the gathered sources."
        ).strip()
    except Exception as exc:
        logger.error("Synthesis call failed: %s", exc, exc_info=True)
        return f"Synthesis failed: {exc}"


# ---------------------------------------------------------------------------
# Private IterDRAG entry point (called directly from ChatEngine dispatch)
# ---------------------------------------------------------------------------


def _deep_research(query: str) -> str:
    """Execute the IterDRAG research loop and return a JSON-serialised result.

    Args:
        query: The research question to investigate.

    Returns:
        JSON string with keys ``answer`` (str), ``citations`` (list of dicts
        with ``url``, ``title``, ``snippet``), and ``iterations`` (int).
    """
    env = _get_env()
    searxng_host: str = env["searxng_host"]
    ollama_model: str = env["ollama_model"]
    max_iterations: int = env["max_iterations"]
    top_n: int = env["top_n"]

    ollama_client = Client(host=env["ollama_host"])

    all_results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    current_query = query
    context_blocks: list[str] = []

    for iteration in range(1, max_iterations + 1):
        logger.info(
            "[research] Iteration %d/%d — query: %r",
            iteration,
            max_iterations,
            current_query,
        )
        try:
            results = _search_searxng(current_query, top_n, searxng_host)
        except (urllib.error.URLError, ValueError) as exc:
            logger.error(
                "[research] SearXNG failed on iteration %d: %s", iteration, exc
            )
            break

        # Deduplicate by URL
        new_results = [r for r in results if r["url"] not in seen_urls]
        for r in new_results:
            seen_urls.add(r["url"])
        all_results.extend(new_results)

        if not results:
            logger.info(
                "[research] No results returned on iteration %d; stopping.", iteration
            )
            break

        block = _format_results_block(results, iteration)
        context_blocks.append(block)

        # Refine the query for the next iteration (skip on last pass)
        if iteration < max_iterations:
            current_query = _refine_query(
                original_query=query,
                context_block="\n\n".join(context_blocks),
                ollama_client=ollama_client,
                model=ollama_model,
            )

    if not all_results:
        return json.dumps(
            {
                "answer": "No results could be retrieved from SearXNG. "
                "Ensure the SearXNG service is running.",
                "citations": [],
                "iterations": 0,
            },
            ensure_ascii=False,
        )

    answer = _synthesise(
        original_query=query,
        all_results=all_results,
        ollama_client=ollama_client,
        model=ollama_model,
    )

    return json.dumps(
        {
            "answer": answer,
            "citations": all_results,
            "iterations": min(max_iterations, len(context_blocks)),
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# FastMCP tool wrapper
# ---------------------------------------------------------------------------


@mcp.tool()
def deep_research(query: str) -> str:
    """Perform iterative deep research and return a synthesised cited answer.

    Uses the IterDRAG methodology: query SearXNG, refine sub-queries via
    Ollama, iterate up to ``RESEARCH_MAX_ITERATIONS`` times, then
    synthesise all gathered evidence into one answer with inline citations.

    Use this tool for questions requiring up-to-date information from
    multiple sources — product releases, recent news, comparative analyses,
    etc.  For simple factual or timeless questions prefer ``web_search``.

    Args:
        query: The research question or topic to investigate.

    Returns:
        JSON string with keys: ``answer`` (synthesised text with inline
        citation markers), ``citations`` (list of source dicts with
        ``url``, ``title``, ``snippet``), and ``iterations`` (int).
    """
    return _deep_research(query=query)
