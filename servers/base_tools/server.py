"""servers/base_tools/server.py

FastMCP server providing base tools for brAIniac chatbot.
Exposes time, real DDG web search, and Open-Meteo weather via MCP protocol.
"""

from __future__ import annotations

# Standard Library
import json
import logging
import urllib.parse
import urllib.request
from typing import Any
from datetime import datetime

# Third-Party Libraries
from ddgs import DDGS
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Initialize FastMCP server with descriptive instructions
mcp: FastMCP = FastMCP(
    "brAIniac-base-tools",
    instructions=(
        "Provides foundational tools for brAIniac: current time, web search, "
        "and current weather for any location."
    ),
)

# WMO Weather Interpretation Codes → human-readable description
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Icy fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight showers",
    81: "Moderate showers",
    82: "Violent showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@mcp.tool()
def get_current_time() -> str:
    """Get the current system date and time.

    Returns:
        JSON string containing the current datetime in ISO 8601 format
        and a human-readable format.
    """
    return _get_current_time()


def _get_current_time() -> str:
    """Implementation: return current time as a JSON string."""
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
    return _web_search(query=query, max_results=max_results)


def _web_search(query: str, max_results: int = 5) -> str:
    """Implementation: run a DuckDuckGo search and return JSON results."""
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


@mcp.tool()
def get_weather(location: str) -> str:
    """Get the current weather conditions for any city or location.

    Uses the Open-Meteo API (no API key required). First resolves the
    location name to coordinates via the Open-Meteo geocoding endpoint,
    then fetches the live current-weather data.

    Args:
        location: City name or location string (e.g. "Seattle", "Paris, FR").

    Returns:
        JSON string with temperature_c, temperature_f, feels_like_c,
        feels_like_f, humidity_pct, wind_speed_kmh, condition, and
        location metadata. Returns an error dict on failure.
    """
    return _get_weather(location)


def _get_weather(location: str) -> str:
    """Implementation: geocode location then fetch Open-Meteo current weather."""
    try:
        # Step 1 — geocode the location name → lat/lon
        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode(
                {"name": location, "count": 1, "language": "en", "format": "json"}
            )
        )
        with urllib.request.urlopen(geo_url, timeout=10) as resp:
            geo_data: dict[str, Any] = json.loads(resp.read())

        results: list[dict[str, Any]] = geo_data.get("results", [])
        if not results:
            return json.dumps({"error": f"Location not found: {location}"})

        geo = results[0]
        lat: float = geo["latitude"]
        lon: float = geo["longitude"]
        resolved_name: str = geo.get("name", location)
        country: str = geo.get("country", "")

        # Step 2 — fetch current weather from Open-Meteo forecast API
        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "current": ",".join(
                        [
                            "temperature_2m",
                            "apparent_temperature",
                            "relative_humidity_2m",
                            "wind_speed_10m",
                            "weather_code",
                            "precipitation",
                        ]
                    ),
                    "temperature_unit": "celsius",
                    "wind_speed_unit": "kmh",
                    "timezone": "auto",
                }
            )
        )
        with urllib.request.urlopen(weather_url, timeout=10) as resp:
            weather_data: dict[str, Any] = json.loads(resp.read())

        current: dict[str, Any] = weather_data.get("current", {})
        temp_c: float = current.get("temperature_2m", 0.0)
        feels_c: float = current.get("apparent_temperature", 0.0)
        humidity: int = current.get("relative_humidity_2m", 0)
        wind_kmh: float = current.get("wind_speed_10m", 0.0)
        precip_mm: float = current.get("precipitation", 0.0)
        wmo_code: int = int(current.get("weather_code", 0))

        condition: str = _WMO_CODES.get(wmo_code, f"Unknown (WMO {wmo_code})")

        output: dict[str, Any] = {
            "location": f"{resolved_name}, {country}".strip(", "),
            "condition": condition,
            "temperature_c": round(temp_c, 1),
            "temperature_f": round(temp_c * 9 / 5 + 32, 1),
            "feels_like_c": round(feels_c, 1),
            "feels_like_f": round(feels_c * 9 / 5 + 32, 1),
            "humidity_pct": humidity,
            "wind_speed_kmh": round(wind_kmh, 1),
            "precipitation_mm": precip_mm,
            "source": "Open-Meteo (open-meteo.com)",
        }
        return json.dumps(output, indent=2)

    except Exception as exc:
        logger.error("[get_weather] Open-Meteo failure: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
