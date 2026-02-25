"""servers/base_tools/server.py

FastMCP server providing base tools for brAIniac chatbot.
Exposes time and mock web search capabilities via MCP protocol.
"""

from __future__ import annotations

# Standard Library
import json
from typing import Any
from datetime import datetime

# Third-Party Libraries
from fastmcp import FastMCP

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
    """Perform a web search for the given query.

    This is a mock implementation for Phase 1. In future phases,
    this will be replaced with a real SearXNG container integration.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        JSON string containing mock search results.
    """
    # Mock search results for Phase 1
    mock_results: list[dict[str, Any]] = [
        {
            "title": f"Mock Result 1 for: {query}",
            "url": "https://example.com/result1",
            "snippet": (
                f"This is a placeholder search result for '{query}'. "
                "In Phase 2, this will be replaced with real SearXNG integration."
            ),
        },
        {
            "title": f"Mock Result 2 for: {query}",
            "url": "https://example.com/result2",
            "snippet": (
                "Another mock result demonstrating the tool interface. "
                "The LLM can parse this structure and provide answers."
            ),
        },
    ]

    # Limit results based on max_results parameter
    limited_results = mock_results[:max_results]

    result: dict[str, Any] = {
        "query": query,
        "results_count": len(limited_results),
        "results": limited_results,
        "note": "Mock implementation - will be replaced with SearXNG in Phase 2",
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
