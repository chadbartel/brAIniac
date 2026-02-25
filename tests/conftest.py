"""tests/conftest.py

Pytest configuration and shared fixtures for the brAIniac test suite.
"""

from __future__ import annotations

# Standard Library
from unittest.mock import Mock

# Third-Party Libraries
import pytest


@pytest.fixture
def mock_ollama_client() -> Mock:
    """Create a mock Ollama client for testing.

    Returns:
        Mock Ollama client with pre-configured responses.
    """
    mock_client = Mock()

    # Mock successful chat response
    mock_client.chat.return_value = {
        "message": {
            "role": "assistant",
            "content": "This is a test response from the mock LLM.",
        },
        "model": "llama3.1:8b-instruct-q4_K_M",
        "created_at": "2026-02-23T00:00:00Z",
        "done": True,
    }

    return mock_client


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Create sample message history for testing.

    Returns:
        List of sample message dictionaries.
    """
    return [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What's the weather like?"},
        {
            "role": "assistant",
            "content": "I don't have real-time weather data, but I can help you find it!",
        },
    ]


@pytest.fixture
def system_message() -> str:
    """Create a sample system message.

    Returns:
        System prompt string.
    """
    return "You are a helpful AI assistant for testing purposes."


@pytest.fixture
def mock_fastmcp_server() -> Mock:
    """Create a mock FastMCP server for testing.

    Returns:
        Mock FastMCP server instance.
    """
    mock_server = Mock()
    mock_server.name = "brAIniac-base-tools"
    return mock_server
