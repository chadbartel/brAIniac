"""
Tools for agent capabilities like web search, data retrieval, etc.
"""

# Standard Library
import os
import logging
from datetime import datetime

# Third Party
import requests

logger = logging.getLogger(__name__)


def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web for current information.

    Tries multiple search APIs in order:
    1. Brave Search API (free tier, no credit card)
    2. DuckDuckGo HTML scraping (fallback)
    3. SearXNG meta-search (if available)

    Args:
        query: The search query
        num_results: Number of results to return (default: 5)

    Returns:
        Formatted search results as a string
    """
    logger.info("Web search called for: %s", query)

    # Try Brave Search API first (free tier available)
    brave_result = _search_brave(query, num_results)
    if brave_result:
        return brave_result

    # Try DuckDuckGo HTML scraping as fallback
    ddg_result = _search_duckduckgo_html(query, num_results)
    if ddg_result:
        return ddg_result

    # If all fails, return error
    logger.error("All search methods failed for query: %s", query)
    return (
        f"❌ Unable to perform web search for: {query}\n\n"
        f"All search methods failed. To fix:\n"
        f"1. Set BRAVE_API_KEY environment variable (free at brave.com/search/api)\n"
        f"2. Or set GOOGLE_API_KEY and GOOGLE_CSE_ID (100 free/day)\n"
        f"3. Or set BING_API_KEY (1000 free/month)\n\n"
        f"Try search_wikipedia() for historical/biographical information."
    )


def _clean_text_for_console(text: str) -> str:
    """
    Remove emojis and special Unicode characters that can't be encoded
    in Windows cp1252.

    Args:
        text: Text potentially containing emojis

    Returns:
        Cleaned text safe for Windows console
    """
    # Remove emojis and other problematic Unicode characters
    return text.encode("ascii", errors="ignore").decode("ascii")


def _search_brave(query: str, num_results: int) -> str:
    """Search using Brave Search API."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        logger.warning("BRAVE_API_KEY not set, falling back to DuckDuckGo")
        return _search_duckduckgo_html(query, num_results)

    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query, "count": num_results}

        response = requests.get(
            url, headers=headers, params=params, timeout=10
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("web", {}).get("results", [])

        if not results:
            logger.info("Brave Search returned no results")
            return "No results found."

        logger.info("Brave Search returned %d results", len(results))

        formatted_results = []
        for i, result in enumerate(results[:num_results], 1):
            title = _clean_text_for_console(result.get("title", ""))
            url_str = result.get("url", "")
            description = _clean_text_for_console(
                result.get("description", "")
            )

            formatted_results.append(
                f"{i}. {title}\n   URL: {url_str}\n   {description}\n"
            )

        return "\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        logger.error("Brave Search API error: %s", e)
        logger.info("Falling back to DuckDuckGo HTML scraping")
        return _search_duckduckgo_html(query, num_results)


def _search_duckduckgo_html(query: str, num_results: int) -> str:
    """
    Search using DuckDuckGo HTML (no API key needed).

    Note: This scrapes HTML and may be unreliable. Use API keys for production.
    """
    try:
        # Third Party
        from bs4 import BeautifulSoup
    except ImportError:
        logger.debug("BeautifulSoup not installed, skipping HTML scraping")
        return ""

    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        data = {"q": query}

        response = requests.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for i, result_div in enumerate(
            soup.find_all("div", class_="result"), 1
        ):
            if i > num_results:
                break

            title_elem = result_div.find("a", class_="result__a")
            snippet_elem = result_div.find("a", class_="result__snippet")

            if title_elem and snippet_elem:
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True)
                link = title_elem.get("href", "")

                results.append(
                    f"{i}. **{title}**\n   {snippet}\n   Source: {link}"
                )

        if results:
            logger.info("DuckDuckGo HTML returned %d results", len(results))
            return f"🔍 Web search results for '{query}':\n\n" + "\n\n".join(
                results
            )

        return ""

    except Exception as e:
        logger.warning("DuckDuckGo HTML search failed: %s", e)
        return ""


def search_news(query: str, days_back: int = 7) -> str:
    """
    Search for recent news articles (simulated - you'd use a real news API).

    Args:
        query: The search query
        days_back: How many days back to search

    Returns:
        Formatted news results
    """
    # In production, use NewsAPI, Google News API, or similar
    # For now, return a message indicating this needs implementation
    return (
        f"News search for '{query}' (last {days_back} days):\n\n"
        "Note: This requires a news API integration. "
        "Consider using:\n"
        "- NewsAPI (https://newsapi.org/)\n"
        "- Google News API\n"
        "- Bing News Search API\n\n"
        "For now, use the general web_search tool which may include "
        "recent information."
    )


def get_current_date() -> str:
    """
    Get the current date and time.

    Returns:
        Current date and time as a string
    """
    now = datetime.now()
    return (
        f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Day of week: {now.strftime('%A')}"
    )


def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for a topic.

    Args:
        query: The search query

    Returns:
        Wikipedia summary
    """
    try:
        # Wikipedia API endpoint
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "format": "json",
            "search": query,
            "limit": 1,
        }

        headers = {
            "User-Agent": "brAIniac/1.0 (Educational Research Assistant)"
        }

        response = requests.get(
            url, params=params, headers=headers, timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if len(data) < 4 or not data[1]:
            return f"No Wikipedia page found for: {query}"

        title = data[1][0] if data[1] else query
        description = data[2][0] if data[2] else "No description available."
        page_url = data[3][0] if data[3] else ""

        return f"**Wikipedia: {title}**\n\n{description}\n\nSource: {page_url}"

    except Exception as e:
        logger.error("Wikipedia search failed: %s", e)
        return f"Error searching Wikipedia: {str(e)}"


# Tool definitions for autogen
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information on any topic. "
                "Use this when you need up-to-date information, news, "
                "or facts about recent events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": (
                "Search Wikipedia for factual information about a topic. "
                "Good for historical facts, biographies, and general "
                "knowledge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to search for",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": (
                "Get the current date and time. Use this to know what "
                "'today' is and to provide context for recent events."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# Map function names to actual functions
TOOL_FUNCTIONS = {
    "search_web": search_web,
    "search_wikipedia": search_wikipedia,
    "get_current_date": get_current_date,
    "search_news": search_news,
}
