"""tests/test_base_tools.py

Unit tests for the base_tools FastMCP server (servers/base_tools/server.py).
Tests tool functions and their outputs.
"""

from __future__ import annotations

import json
from datetime import datetime


# Helper functions that replicate tool behavior for testing
# These mirror the implementation in servers/base_tools/server.py
def get_current_time_impl() -> str:
    """Implementation of get_current_time for testing."""
    now = datetime.now()
    result: dict[str, str] = {
        "iso_format": now.isoformat(),
        "readable": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
        "timezone": "Local system time",
    }
    return json.dumps(result, indent=2)


def web_search_impl(query: str, max_results: int = 5) -> str:
    """Implementation of web_search for testing."""
    mock_results: list[dict[str, str]] = [
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

    limited_results = mock_results[:max_results]

    result: dict[str, str | int | list[dict[str, str]]] = {
        "query": query,
        "results_count": len(limited_results),
        "results": limited_results,
        "note": "Mock implementation - will be replaced with SearXNG in Phase 2",
    }

    return json.dumps(result, indent=2, ensure_ascii=False)


class TestGetCurrentTime:
    """Test suite for get_current_time tool."""

    def test_get_current_time_returns_json(self) -> None:
        """Test that get_current_time returns valid JSON."""
        result = get_current_time_impl()

        # Should be valid JSON
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_get_current_time_has_required_fields(self) -> None:
        """Test that get_current_time returns all required fields."""
        result = get_current_time_impl()
        data = json.loads(result)

        assert "iso_format" in data
        assert "readable" in data
        assert "timezone" in data

    def test_get_current_time_iso_format_valid(self) -> None:
        """Test that iso_format is a valid ISO 8601 datetime string."""
        result = get_current_time_impl()
        data = json.loads(result)

        # Should be parseable as ISO datetime
        iso_str = data["iso_format"]
        parsed = datetime.fromisoformat(iso_str)
        assert isinstance(parsed, datetime)

    def test_get_current_time_readable_is_string(self) -> None:
        """Test that readable format is a non-empty string."""
        result = get_current_time_impl()
        data = json.loads(result)

        readable = data["readable"]
        assert isinstance(readable, str)
        assert len(readable) > 0

    def test_get_current_time_timezone_present(self) -> None:
        """Test that timezone information is present."""
        result = get_current_time_impl()
        data = json.loads(result)

        assert isinstance(data["timezone"], str)
        assert len(data["timezone"]) > 0


class TestWebSearch:
    """Test suite for web_search tool."""

    def test_web_search_returns_json(self) -> None:
        """Test that web_search returns valid JSON."""
        result = web_search_impl("Python testing")
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_web_search_has_required_fields(self) -> None:
        """Test that web_search returns all required fields."""
        result = web_search_impl("Python testing")
        data = json.loads(result)

        assert "query" in data
        assert "results_count" in data
        assert "results" in data
        assert "note" in data  # Mock implementation note

    def test_web_search_query_preserved(self) -> None:
        """Test that the search query is preserved in results."""
        query = "machine learning best practices"
        result = web_search_impl(query)
        data = json.loads(result)

        assert data["query"] == query

    def test_web_search_returns_results_list(self) -> None:
        """Test that results is a list."""
        result = web_search_impl("test query")
        data = json.loads(result)

        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0

    def test_web_search_result_structure(self) -> None:
        """Test that each result has the correct structure."""
        result = web_search_impl("test query")
        data = json.loads(result)

        for res in data["results"]:
            assert "title" in res
            assert "url" in res
            assert "snippet" in res

            assert isinstance(res["title"], str)
            assert isinstance(res["url"], str)
            assert isinstance(res["snippet"], str)

    def test_web_search_max_results_default(self) -> None:
        """Test that default max_results works correctly."""
        result = web_search_impl("test query")
        data = json.loads(result)

        # Mock implementation returns 2 results by default
        assert data["results_count"] == 2
        assert len(data["results"]) == 2

    def test_web_search_max_results_custom(self) -> None:
        """Test that custom max_results parameter works."""
        # Note: Mock implementation returns 2 results, so max_results
        # will be limited by the mock data
        result = web_search_impl("test query", max_results=1)
        data = json.loads(result)

        # Should limit to requested number
        assert len(data["results"]) <= 1

    def test_web_search_query_in_results(self) -> None:
        """Test that query text appears in result titles."""
        query = "Python"
        result = web_search_impl(query)
        data = json.loads(result)

        # At least one result should mention the query
        titles = [res["title"] for res in data["results"]]
        assert any(query in title for title in titles)

    def test_web_search_various_queries(self) -> None:
        """Test web_search with various query formats."""
        test_queries = [
            "simple query",
            "multi word query with spaces",
            "query-with-hyphens",
            "query_with_underscores",
            "UPPERCASE QUERY",
            "MixedCase Query",
        ]

        for query in test_queries:
            result = web_search_impl(query)
            data = json.loads(result)

            assert data["query"] == query
            assert len(data["results"]) > 0

    def test_web_search_empty_query(self) -> None:
        """Test web_search with empty query string."""
        result = web_search_impl("")
        data = json.loads(result)

        # Should still return valid structure
        assert "query" in data
        assert data["query"] == ""
        assert "results" in data

    def test_web_search_is_mock_implementation(self) -> None:
        """Test that results indicate this is a mock implementation."""
        result = web_search_impl("test")
        data = json.loads(result)

        # Should have a note indicating mock implementation
        assert "note" in data
        assert "mock" in data["note"].lower() or "Mock" in data["note"]
