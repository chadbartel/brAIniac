"""tests/test_base_tools.py

Unit tests for the base_tools FastMCP server (servers/base_tools/server.py).
Imports and tests the *actual* tool functions; DDG and HTTP are mocked out so
these tests are fully offline and deterministic.
"""

from __future__ import annotations

# Standard Library
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

# Third-Party Libraries
import pytest

# Local Modules — unwrap FastMCP's FunctionTool to get the plain Python callables
from servers.base_tools.server import (
    get_current_time as _get_current_time_tool,
    get_weather as _get_weather_tool,
    web_search as _web_search_tool,
)

# FunctionTool wraps the original function at `.fn`; import those for direct testing.
get_current_time = _get_current_time_tool.fn
web_search = _web_search_tool.fn
get_weather = _get_weather_tool.fn

# ---------------------------------------------------------------------------
# Shared DDG mock factory
# ---------------------------------------------------------------------------

_MOCK_DDG_HITS: list[dict[str, str]] = [
    {"title": "Result A", "href": "https://example.com/a", "body": "Snippet A"},
    {"title": "Result B", "href": "https://example.com/b", "body": "Snippet B"},
    {"title": "Result C", "href": "https://example.com/c", "body": "Snippet C"},
]


def _ddgs_patch(hits: list[dict[str, str]] | None = None) -> MagicMock:
    """Return a MagicMock that behaves as the DDGS context manager.

    Args:
        hits: Fake search hit dicts to return from ``ddgs.text()``.
              Defaults to ``_MOCK_DDG_HITS``.

    Returns:
        A mock suitable for use with ``patch("servers.base_tools.server.DDGS", ...)``.
    """
    mock_cls = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.text.return_value = hits if hits is not None else _MOCK_DDG_HITS
    mock_cls.return_value.__enter__.return_value = mock_ctx
    mock_cls.return_value.__exit__.return_value = False
    return mock_cls


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------


class TestGetCurrentTime:
    """Test suite for the get_current_time tool."""

    def test_returns_valid_json(self) -> None:
        """get_current_time must return a valid JSON string."""
        result = get_current_time()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_has_required_fields(self) -> None:
        """Returned JSON must contain iso_format, readable, and timezone."""
        data = json.loads(get_current_time())
        assert "iso_format" in data
        assert "readable" in data
        assert "timezone" in data

    def test_iso_format_is_parseable(self) -> None:
        """iso_format must be a valid ISO 8601 datetime string."""
        data = json.loads(get_current_time())
        parsed = datetime.fromisoformat(data["iso_format"])
        assert isinstance(parsed, datetime)

    def test_readable_is_non_empty_string(self) -> None:
        """readable must be a non-empty string."""
        data = json.loads(get_current_time())
        assert isinstance(data["readable"], str)
        assert len(data["readable"]) > 0

    def test_timezone_is_non_empty_string(self) -> None:
        """timezone must be a non-empty string."""
        data = json.loads(get_current_time())
        assert isinstance(data["timezone"], str)
        assert len(data["timezone"]) > 0

    def test_iso_format_reflects_current_time(self) -> None:
        """iso_format must be close to the actual current time (within 5 s)."""
        before = datetime.now()
        data = json.loads(get_current_time())
        after = datetime.now()
        parsed = datetime.fromisoformat(data["iso_format"])
        assert before <= parsed <= after


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


class TestWebSearch:
    """Test suite for the web_search tool (DDG mocked)."""

    def test_returns_valid_json(self) -> None:
        """web_search must return a valid JSON string."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            result = web_search("Python testing")
        assert isinstance(json.loads(result), dict)

    def test_has_required_fields(self) -> None:
        """Returned JSON must contain query, results_count, and results."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search("Python testing"))
        assert "query" in data
        assert "results_count" in data
        assert "results" in data

    def test_query_is_preserved(self) -> None:
        """The query field must echo the input query string."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search("machine learning best practices"))
        assert data["query"] == "machine learning best practices"

    def test_results_is_a_list(self) -> None:
        """results must be a non-empty list."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search("test query"))
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0

    def test_results_count_matches_list_length(self) -> None:
        """results_count must equal len(results)."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search("test"))
        assert data["results_count"] == len(data["results"])

    def test_result_items_have_correct_keys(self) -> None:
        """Each result must have title, url, and snippet string fields."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search("test query"))
        for item in data["results"]:
            assert isinstance(item.get("title"), str)
            assert isinstance(item.get("url"), str)
            assert isinstance(item.get("snippet"), str)

    def test_max_results_forwarded_to_ddg(self) -> None:
        """max_results must be forwarded verbatim to DDGS.text()."""
        mock_cls = _ddgs_patch(hits=_MOCK_DDG_HITS[:2])
        with patch("servers.base_tools.server.DDGS", mock_cls):
            web_search("test", max_results=2)
        ctx = mock_cls.return_value.__enter__.return_value
        ctx.text.assert_called_once_with("test", max_results=2)

    def test_href_mapped_to_url(self) -> None:
        """DDG returns 'href'; server must remap it to 'url'."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search("test"))
        assert data["results"][0]["url"] == "https://example.com/a"

    def test_body_mapped_to_snippet(self) -> None:
        """DDG returns 'body'; server must remap it to 'snippet'."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search("test"))
        assert data["results"][0]["snippet"] == "Snippet A"

    def test_ddg_exception_returns_error_json(self) -> None:
        """A DDG failure must return JSON with an 'error' key (no exception raised)."""
        mock_cls = _ddgs_patch()
        mock_cls.return_value.__enter__.return_value.text.side_effect = RuntimeError(
            "DDG connection failed"
        )
        with patch("servers.base_tools.server.DDGS", mock_cls):
            result = web_search("test")
        data = json.loads(result)
        assert "error" in data
        assert "DDG connection failed" in data["error"]

    @pytest.mark.parametrize(
        "query",
        [
            "simple query",
            "multi word query with spaces",
            "query-with-hyphens",
            "query_with_underscores",
            "UPPERCASE QUERY",
            "MixedCase Query",
            "",
        ],
    )
    def test_various_query_formats(self, query: str) -> None:
        """web_search must preserve the query string regardless of format."""
        with patch("servers.base_tools.server.DDGS", _ddgs_patch()):
            data = json.loads(web_search(query))
        assert data["query"] == query


# ---------------------------------------------------------------------------
# get_weather
# ---------------------------------------------------------------------------

# Reusable fake API payloads
_FAKE_GEO_RESPONSE = json.dumps({
    "results": [
        {
            "name": "Seattle",
            "latitude": 47.6062,
            "longitude": -122.3321,
            "country": "United States",
        }
    ]
}).encode()

_FAKE_WEATHER_RESPONSE = json.dumps({
    "current": {
        "temperature_2m": 12.5,
        "apparent_temperature": 10.0,
        "relative_humidity_2m": 78,
        "wind_speed_10m": 14.4,
        "weather_code": 3,
        "precipitation": 0.0,
    }
}).encode()

_FAKE_EMPTY_GEO_RESPONSE = json.dumps({"results": []}).encode()


def _make_urlopen_mock(responses: list[bytes]) -> MagicMock:
    """Return a urlopen mock that yields successive byte payloads.

    Each call to urlopen().__enter__().read() returns the next item in
    *responses*, so the first call gets responses[0], the second gets
    responses[1], etc.
    """
    call_count = 0

    def _side_effect(url: str, timeout: int = 10) -> MagicMock:
        nonlocal call_count
        payload = responses[call_count]
        call_count += 1
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.read.return_value = payload
        return ctx

    mock = MagicMock(side_effect=_side_effect)
    return mock


class TestGetWeather:
    """Test suite for the get_weather Open-Meteo tool (HTTP mocked)."""

    def test_returns_valid_json(self) -> None:
        """get_weather must return a valid JSON string."""
        mock_urlopen = _make_urlopen_mock([_FAKE_GEO_RESPONSE, _FAKE_WEATHER_RESPONSE])
        with patch("servers.base_tools.server.urllib.request.urlopen", mock_urlopen):
            result = get_weather("Seattle")
        assert isinstance(json.loads(result), dict)

    def test_has_required_fields(self) -> None:
        """Response must contain all expected weather fields."""
        mock_urlopen = _make_urlopen_mock([_FAKE_GEO_RESPONSE, _FAKE_WEATHER_RESPONSE])
        with patch("servers.base_tools.server.urllib.request.urlopen", mock_urlopen):
            data = json.loads(get_weather("Seattle"))
        for field in (
            "location", "condition", "temperature_c", "temperature_f",
            "feels_like_c", "feels_like_f", "humidity_pct",
            "wind_speed_kmh", "precipitation_mm", "source",
        ):
            assert field in data, f"Missing field: {field}"

    def test_temperature_conversion(self) -> None:
        """Fahrenheit values must be correctly converted from Celsius."""
        mock_urlopen = _make_urlopen_mock([_FAKE_GEO_RESPONSE, _FAKE_WEATHER_RESPONSE])
        with patch("servers.base_tools.server.urllib.request.urlopen", mock_urlopen):
            data = json.loads(get_weather("Seattle"))
        expected_f = round(12.5 * 9 / 5 + 32, 1)
        assert data["temperature_f"] == expected_f

    def test_resolved_location_in_response(self) -> None:
        """The resolved city name from the geocoder must appear in 'location'."""
        mock_urlopen = _make_urlopen_mock([_FAKE_GEO_RESPONSE, _FAKE_WEATHER_RESPONSE])
        with patch("servers.base_tools.server.urllib.request.urlopen", mock_urlopen):
            data = json.loads(get_weather("Seattle"))
        assert "Seattle" in data["location"]

    def test_wmo_code_resolved_to_condition_string(self) -> None:
        """WMO code 3 must resolve to 'Overcast'."""
        mock_urlopen = _make_urlopen_mock([_FAKE_GEO_RESPONSE, _FAKE_WEATHER_RESPONSE])
        with patch("servers.base_tools.server.urllib.request.urlopen", mock_urlopen):
            data = json.loads(get_weather("Seattle"))
        assert data["condition"] == "Overcast"

    def test_unknown_location_returns_error(self) -> None:
        """When geocoding returns no results, an error dict must be returned."""
        mock_urlopen = _make_urlopen_mock([_FAKE_EMPTY_GEO_RESPONSE])
        with patch("servers.base_tools.server.urllib.request.urlopen", mock_urlopen):
            data = json.loads(get_weather("ZZZNowhere123"))
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_network_error_returns_error_json(self) -> None:
        """Any network exception must return a JSON error dict, not raise."""
        with patch(
            "servers.base_tools.server.urllib.request.urlopen",
            side_effect=OSError("Network unreachable"),
        ):
            data = json.loads(get_weather("Seattle"))
        assert "error" in data
        assert "Network unreachable" in data["error"]

    def test_source_field_credits_open_meteo(self) -> None:
        """The 'source' field must reference Open-Meteo."""
        mock_urlopen = _make_urlopen_mock([_FAKE_GEO_RESPONSE, _FAKE_WEATHER_RESPONSE])
        with patch("servers.base_tools.server.urllib.request.urlopen", mock_urlopen):
            data = json.loads(get_weather("Seattle"))
        assert "open-meteo" in data["source"].lower()
