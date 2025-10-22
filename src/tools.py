"""
Tools for agent capabilities like web search, data retrieval, etc.
"""

# Standard Library
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# Third Party
import requests

logger = logging.getLogger(__name__)


def search_web(query: str, num_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo API (free, no API key needed).

    Args:
        query: The search query
        num_results: Number of results to return

    Returns:
        Formatted search results as a string
    """
    try:
        # DuckDuckGo Instant Answer API (free, no auth)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []

        # Get abstract if available
        if data.get("Abstract"):
            results.append(
                f"**Overview**: {data['Abstract']}\n"
                f"Source: {data.get('AbstractURL', 'N/A')}"
            )

        # Get related topics
        if data.get("RelatedTopics"):
            for i, topic in enumerate(data["RelatedTopics"][:num_results], 1):
                if isinstance(topic, dict) and "Text" in topic:
                    text = topic.get("Text", "")
                    url = topic.get("FirstURL", "")
                    results.append(f"{i}. {text}\n   Source: {url}")

        if not results:
            return f"No results found for query: {query}"

        return f"Search results for '{query}':\n\n" + "\n\n".join(results)

    except Exception as e:
        logger.error("Web search failed: %s", e)
        return f"Error performing web search: {str(e)}"


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
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "search": query,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return f"No Wikipedia page found for: {query}"

        # Get the first page
        page = next(iter(pages.values()))

        if "missing" in page:
            return f"No Wikipedia page found for: {query}"

        title = page.get("title", query)
        extract = page.get("extract", "No summary available.")

        return (
            f"**Wikipedia: {title}**\n\n"
            f"{extract}\n\n"
            f"Source: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        )

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
