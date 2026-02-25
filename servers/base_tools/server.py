"""servers/base_tools/server.py

FastMCP server providing base tools for brAIniac chatbot.
Exposes time and real DDG web search capabilities via MCP protocol.
"""

from __future__ import annotations

# Standard Library
import json
import logging
from datetime import datetime
from typing import Any

# Third-Party Libraries
from ddgs import DDGS
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize FastMCP server with descriptive instructions
mcp: FastMCP = FastMCP(
    "brAIniac-base-tools",
    instructions=(
        "Provides foundational tools for brAIniac: current time and web search."
    ),
)


@mcp.tool()
def get_current_time() -> str:
    """Get the current system date and time.

    Returns:
        JSON string containing the current datetime in ISO 8601 format
        and a human-readable format.
    """
    now = datetime.now()
    result: dict[str, str] = {
        "iso_format": now.isoformat(),
        "readable": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
        "timezone": "Local system time",
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current real-world information.

    Uses DuckDuckGo (DDG) in-process — no API key required.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        JSON string containing search results with title, url, and snippet
        fields, or an error dict on failure.
    """
    try:
        with DDGS() as ddgs:
            hits = [
                {"title": r["title"], "url": r["href"], "snippet": r["body"]}
                for r in ddgs.text(query, max_results=max_results)
            ]
        result: dict[str, Any] = {
            "query": query,
            "results_count": len(hits),
            "results": hits,
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("[web_search] DDG failure: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
