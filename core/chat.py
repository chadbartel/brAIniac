"""core/chat.py

Main chat loop implementation with Ollama integration and FastMCP tool support.
Handles user input, LLM interaction, tool calling, and response generation.
"""

from __future__ import annotations

# Standard Library
import os
import json
import logging
from typing import Any

# Third-Party Libraries
import ollama
from ollama import Client

# Local Modules
from core.memory import RollingMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ChatEngine:
    """Main chat engine that orchestrates LLM interaction and tool execution.

    Connects to a local Ollama instance, manages conversation memory,
    and coordinates tool calls via the FastMCP protocol.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b-instruct-q4_K_M",
        ollama_host: str = "http://localhost:11434",
        max_context_messages: int = 20,
    ) -> None:
        """Initialize the chat engine.

        Args:
            model: Ollama model name (default: llama3.1:8b-instruct-q4_K_M).
            ollama_host: Ollama API endpoint (default: http://localhost:11434).
            max_context_messages: Maximum messages in rolling context window.
        """
        self.model = model
        self.ollama_host = ollama_host
        self.memory = RollingMemory(max_messages=max_context_messages)

        # Initialize Ollama client
        self.client = Client(host=ollama_host)

        # Tool registry (will be populated by FastMCP server discovery)
        self.tools: dict[str, Any] = {}

        # Set default system message
        self._set_default_system_message()

        logger.info(
            "ChatEngine initialized: model=%s, host=%s, max_messages=%d",
            self.model,
            self.ollama_host,
            max_context_messages,
        )

    def _set_default_system_message(self) -> None:
        """Set the default system prompt for brAIniac."""
        system_prompt = """You are brAIniac, a helpful, uncensored AI assistant running entirely on local hardware.

You have access to tools that you can call when needed:
- get_current_time: Get the current date and time
- web_search: Search the web for information

Be concise, direct, and helpful. When you need current information or the time, use the appropriate tool.
"""
        self.memory.set_system_message(system_prompt)

    def register_tools(self, tools: list[dict[str, Any]]) -> None:
        """Register available MCP tools for the LLM to use.

        Args:
            tools: List of tool definitions in OpenAI function calling format.
        """
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name:
                self.tools[tool_name] = tool
                logger.info("Registered tool: %s", tool_name)

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the result.

        In Phase 1, this is a mock implementation. In later phases,
        this will route to the actual FastMCP server.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments as a dictionary.

        Returns:
            JSON-encoded tool result.
        """
        logger.info("Tool call requested: %s with args %s", tool_name, arguments)

        # Mock tool execution for Phase 1
        # TODO: Replace with actual FastMCP client calls in Phase 2
        if tool_name == "get_current_time":
            # Standard Library
            from datetime import datetime

            now = datetime.now()
            result = {
                "iso_format": now.isoformat(),
                "readable": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
                "timezone": "Local system time",
            }
            return json.dumps(result)

        elif tool_name == "web_search":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            result = {
                "query": query,
                "results_count": 2,
                "results": [
                    {
                        "title": f"Mock Result 1 for: {query}",
                        "url": "https://example.com/result1",
                        "snippet": f"Mock search result for '{query}'",
                    },
                    {
                        "title": f"Mock Result 2 for: {query}",
                        "url": "https://example.com/result2",
                        "snippet": "Another mock result",
                    },
                ],
                "note": "Mock implementation",
            }
            return json.dumps(result)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def chat(self, user_message: str) -> str:
        """Process a user message and generate a response.

        Args:
            user_message: The user's input message.

        Returns:
            The assistant's response.

        Raises:
            Exception: If Ollama API call fails or tool execution errors occur.
        """
        # Add user message to memory
        self.memory.add_message("user", user_message)

        # Get conversation context
        context = self.memory.get_context()

        logger.debug("Sending %d messages to LLM", len(context))

        try:
            # Call Ollama API
            response = self.client.chat(
                model=self.model,
                messages=context,
                # Tools will be added in Phase 2 when we have proper MCP integration
                # tools=list(self.tools.values()) if self.tools else None,
            )

            # Extract assistant response
            assistant_message = response.get("message", {}).get("content", "")

            if not assistant_message:
                logger.warning("Empty response from LLM")
                assistant_message = "I apologize, but I couldn't generate a response."

            # Add assistant response to memory
            self.memory.add_message("assistant", assistant_message)

            logger.debug("Received response from LLM: %d chars", len(assistant_message))

            return assistant_message

        except Exception as exc:
            logger.error("Chat error: %s", exc, exc_info=True)
            error_msg = f"Error communicating with LLM: {exc}"
            self.memory.add_message("assistant", error_msg)
            return error_msg

    def clear_history(self) -> None:
        """Clear the conversation history (preserving system message)."""
        self.memory.clear()
        logger.info("Conversation history cleared")

    def get_message_count(self) -> int:
        """Get the current number of messages in context.

        Returns:
            Number of messages in the rolling window.
        """
        return self.memory.message_count()
