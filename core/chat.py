"""core/chat.py

Main chat loop implementation with Ollama integration and FastMCP tool support.
Handles user input, LLM interaction, tool calling, and response generation.
"""

from __future__ import annotations

# Standard Library
import os
import json
import logging
from typing import Any, Literal, Callable
from pathlib import Path

# Third-Party Libraries
from ollama import Client

# Local Modules
from core.memory import BaseMemory, DiskMemory, RollingMemory
from core.personality import PersonalityManager, PersonalityVectors
from core.intent_classifier import needs_current_information

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
        model: str | None = None,
        ollama_host: str | None = None,
        max_context_messages: int = 20,
        personality_vectors: PersonalityVectors | None = None,
        memory_backend: Literal["rolling", "disk"] = "rolling",
        persist_path: Path | None = None,
    ) -> None:
        """Initialize the chat engine.

        Args:
            model: Ollama model name. Falls back to the OLLAMA_MODEL env var,
                then "llama3.1:8b-instruct-q4_K_M".
            ollama_host: Ollama API endpoint. Falls back to the OLLAMA_HOST
                env var, then "http://localhost:11434".
            max_context_messages: Maximum messages in rolling context window.
            personality_vectors: Pre-built PersonalityVectors instance. When
                omitted, vectors are loaded from env vars via
                ``PersonalityVectors.from_env()``.
            memory_backend: Which memory implementation to use.
                ``"rolling"`` (default) keeps all existing behaviour;
                ``"disk"`` activates the Phase 2 JSONL-backed DiskMemory.
            persist_path: Custom path for the DiskMemory JSONL file.
                Forwarded verbatim to DiskMemory; ignored for rolling backend.
        """
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
        self.ollama_host = ollama_host or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        )

        self.memory: BaseMemory
        if memory_backend == "disk":
            self.memory = DiskMemory(
                max_messages=max_context_messages, persist_path=persist_path
            )
        else:
            self.memory = RollingMemory(max_messages=max_context_messages)

        # Initialize Ollama client
        self.client = Client(host=self.ollama_host)

        # Tool registry + dispatch table
        self.tools: dict[str, Any] = {}
        self._tool_dispatch: dict[str, Callable[[dict[str, Any]], str]] = {}

        # Initialize PersonalityManager and inject the dynamic system prompt
        vectors = personality_vectors or PersonalityVectors.from_env()
        self.personality_manager = PersonalityManager(vectors)
        self._set_default_system_message()

        # Register built-in tool schemas and wire up the dispatch table
        self._register_default_tools()
        self._build_tool_dispatch()

        logger.info(
            "ChatEngine initialized: model=%s, host=%s, max_messages=%d, "
            "personality=snark=%.1f/verbosity=%.1f/empathy=%.1f",
            self.model,
            self.ollama_host,
            max_context_messages,
            vectors.snark,
            vectors.verbosity,
            vectors.empathy,
        )

    def _set_default_system_message(self) -> None:
        """Set the system prompt by delegating to PersonalityManager."""
        system_prompt = self.personality_manager.generate_system_prompt()
        self.memory.set_system_message(system_prompt)

    def _build_tool_dispatch(self) -> None:
        """Build the tool-name → executor mapping.

        Using a dict registry instead of an if/elif chain makes it trivial
        to extend the dispatch table (add one entry) without touching the
        ``execute_tool`` method body.
        """

        def _exec_get_current_time(args: dict[str, Any]) -> str:
            # Local Modules
            from servers.base_tools.server import _get_current_time

            return _get_current_time()

        def _exec_web_search(args: dict[str, Any]) -> str:
            # Local Modules
            from servers.base_tools.server import _web_search

            return _web_search(
                query=args.get("query", ""),
                max_results=int(args.get("max_results", 5)),
            )

        def _exec_get_weather(args: dict[str, Any]) -> str:
            # Local Modules
            from servers.base_tools.server import _get_weather

            return _get_weather(location=args.get("location", ""))

        def _exec_deep_research(args: dict[str, Any]) -> str:
            # Local Modules
            from servers.research_server.server import _deep_research

            return _deep_research(query=args.get("query", ""))

        self._tool_dispatch = {
            "get_current_time": _exec_get_current_time,
            "web_search": _exec_web_search,
            "get_weather": _exec_get_weather,
            "deep_research": _exec_deep_research,
        }

    def _register_default_tools(self) -> None:
        """Register the built-in tool schemas (Phase 1 + Phase 2) with the engine."""
        self.register_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "Get the current local date and time.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": (
                            "Search the web for current, real-world information such as "
                            "weather, news, prices, sports scores, and general facts."
                        ),
                        "parameters": {
                            "type": "object",
                            "required": ["query"],
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string.",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results to return (default 5).",
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": (
                            "Get the current weather conditions for any city or location. "
                            "Use this for questions about current weather, temperature, "
                            "humidity, wind, or forecast."
                        ),
                        "parameters": {
                            "type": "object",
                            "required": ["location"],
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name or location string (e.g. 'Seattle' or 'Paris, FR').",
                                },
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "deep_research",
                        "description": (
                            "Perform iterative deep research on a topic by querying multiple "
                            "web sources, refining sub-queries, and synthesising a cited answer. "
                            "Use this for complex questions that require up-to-date information "
                            "from multiple sources, such as recent product releases, news events, "
                            "or comparative analyses."
                        ),
                        "parameters": {
                            "type": "object",
                            "required": ["query"],
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The research question or topic to investigate.",
                                },
                            },
                        },
                    },
                },
            ]
        )

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

        Dispatches via ``_tool_dispatch`` dict — extend that table to add
        new tools without modifying this method.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments as a dictionary.

        Returns:
            JSON-encoded tool result.
        """
        logger.info("Tool call requested: %s with args %s", tool_name, arguments)

        executor = self._tool_dispatch.get(tool_name)
        if executor is not None:
            return executor(arguments)

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

        # --- Intent gating ---------------------------------------------------
        # Run the DistilBERT zero-shot classifier to decide whether research
        # tools should be offered to the LLM on this turn.  Base tools
        # (time, weather, web_search) are always included.  deep_research is
        # only appended when the classifier flags a temporal information need.
        research_needed: bool = needs_current_information(user_message)
        _research_tool_names: frozenset[str] = frozenset({"deep_research"})
        turn_schemas: list[dict[str, Any]] = [
            schema
            for name, schema in self.tools.items()
            if name not in _research_tool_names or research_needed
        ]
        tool_schemas: list[dict[str, Any]] | None = (
            turn_schemas if turn_schemas else None
        )

        logger.debug(
            "Turn tool schemas: %s (research_needed=%s)",
            [s["function"]["name"] for s in (tool_schemas or [])],
            research_needed,
        )

        # Tool-call exchanges are kept in a local buffer for this turn only.
        # They are not persisted to rolling memory to avoid context bloat.
        turn_messages: list[dict[str, Any]] = list(context)

        try:
            response = self.client.chat(
                model=self.model,
                messages=turn_messages,
                tools=tool_schemas,
            )

            # Agentic tool-call dispatch loop.
            # If the model requests one or more tools, execute them and feed
            # the results back until the model produces a plain text reply.
            while True:
                raw_msg = response["message"]
                # getattr supports real Ollama Pydantic models (.tool_calls attr);
                # plain dict mocks will return None (no such attribute) safely.
                tool_calls = getattr(raw_msg, "tool_calls", None) or []

                if not tool_calls:
                    break

                # Append the assistant's tool-call turn to the local buffer
                turn_messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                # Execute each tool and add results to the local buffer
                for tc in tool_calls:
                    tool_result = self.execute_tool(
                        tc.function.name,
                        dict(tc.function.arguments) if tc.function.arguments else {},
                    )
                    logger.info(
                        "Tool executed: %s → %s", tc.function.name, tool_result[:200]
                    )
                    turn_messages.append({"role": "tool", "content": tool_result})

                # Ask the model to continue now that it has the tool results
                response = self.client.chat(
                    model=self.model,
                    messages=turn_messages,
                    tools=tool_schemas,
                )

            # Extract the final plain-text reply
            raw_msg = response["message"]
            assistant_message = (
                getattr(raw_msg, "content", None)
                or (raw_msg.get("content", "") if isinstance(raw_msg, dict) else "")
                or ""
            )

            if not assistant_message:
                logger.warning("Empty response from LLM")
                assistant_message = "I apologize, but I couldn't generate a response."

            # Persist only the final reply to rolling memory
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
