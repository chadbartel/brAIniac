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

    Note: DuckDuckGo's free API is limited. For production use:
    - Google Custom Search API
    - Bing Web Search API
    - SerpAPI

    Args:
        query: The search query
        num_results: Number of results to return

    Returns:
        Formatted search results as a string
    """
    logger.info("Web search called for: %s", query)

    try:
        # DuckDuckGo Instant Answer API (free, no auth)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        headers = {
            "User-Agent": "brAIniac/1.0 (Educational Research Assistant)"
        }

        response = requests.get(
            url, params=params, headers=headers, timeout=10
        )
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
                    topic_url = topic.get("FirstURL", "")
                    results.append(f"{i}. {text}\n   Source: {topic_url}")

        if results:
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)

        # If no results from DuckDuckGo, return a helpful message
        return (
            f"⚠️ No results from DuckDuckGo for: {query}\n\n"
            f"DuckDuckGo's free API has limited coverage. "
            f"For production use, integrate:\n"
            f"- Google Custom Search API (100 free queries/day)\n"
            f"- Bing Web Search API (1000 free queries/month)\n"
            f"- SerpAPI (paid, comprehensive)\n\n"
            f"Try using search_wikipedia() for factual/historical information."
        )

    except Exception as e:
        logger.error("Web search failed: %s", e)
        return (
            f"⚠️ Web search error: {str(e)}\n\n"
            f"Try using search_wikipedia() for factual information instead."
        )


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

        return (
            f"**Wikipedia: {title}**\n\n"
            f"{description}\n\n"
            f"Source: {page_url}"
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
