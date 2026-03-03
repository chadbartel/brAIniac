"""tests/test_chat.py

Unit tests for the ChatEngine class (core/chat.py).
Tests chat loop, Ollama integration (mocked), and tool execution.
"""

from __future__ import annotations

# Standard Library
from pathlib import Path
from unittest.mock import Mock, patch

# Local Modules
from core.chat import ChatEngine
from core.memory import DiskMemory, RollingMemory


class TestChatEngine:
    """Test suite for ChatEngine class."""

    @patch("core.chat.Client")
    def test_initialization_default(self, mock_client_class: Mock) -> None:
        """Test ChatEngine initializes with default parameters."""
        engine = ChatEngine()

        assert engine.model == "llama3.1:8b-instruct-q4_K_M"
        assert engine.ollama_host == "http://localhost:11434"
        assert engine.memory.max_messages == 20
        # Default tools are registered on init
        assert "get_current_time" in engine.tools
        assert "web_search" in engine.tools

        # Verify Ollama client was initialized
        mock_client_class.assert_called_once_with(host="http://localhost:11434")

    @patch("core.chat.Client")
    def test_initialization_custom_parameters(self, mock_client_class: Mock) -> None:
        """Test ChatEngine initializes with custom parameters."""
        custom_model = "qwen2.5:7b-instruct-q4_K_M"
        custom_host = "http://custom-host:11434"
        custom_max_messages = 10

        engine = ChatEngine(
            model=custom_model,
            ollama_host=custom_host,
            max_context_messages=custom_max_messages,
        )

        assert engine.model == custom_model
        assert engine.ollama_host == custom_host
        assert engine.memory.max_messages == custom_max_messages

        mock_client_class.assert_called_once_with(host=custom_host)

    @patch("core.chat.Client")
    def test_system_message_set_on_init(self, mock_client_class: Mock) -> None:
        """Test that default system message is set during initialization."""
        engine = ChatEngine()

        context = engine.memory.get_context()
        assert len(context) == 1
        assert context[0]["role"] == "system"
        assert "brAIniac" in context[0]["content"]

    @patch("core.chat.Client")
    def test_register_tools(self, mock_client_class: Mock) -> None:
        """Test registering MCP tools."""
        engine = ChatEngine()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current time",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"query": {"type": "string"}},
                },
            },
        ]

        engine.register_tools(tools)

        assert (
            len(engine.tools) == 4
        )  # 2 passed + get_weather + deep_research registered at init
        assert "get_current_time" in engine.tools
        assert "web_search" in engine.tools

    @patch("core.chat.Client")
    def test_execute_tool_get_current_time(self, mock_client_class: Mock) -> None:
        """Test executing the get_current_time tool."""
        engine = ChatEngine()
        result = engine.execute_tool("get_current_time", {})

        # Should return JSON string
        # Standard Library
        import json

        data = json.loads(result)
        assert "iso_format" in data
        assert "readable" in data
        assert "timezone" in data

    @patch("core.chat.Client")
    def test_execute_tool_web_search(self, mock_client_class: Mock) -> None:
        """Test executing the web_search tool calls the real server function."""
        engine = ChatEngine()

        # Standard Library
        import json

        fake_result = json.dumps(
            {
                "query": "Python testing",
                "results_count": 1,
                "results": [
                    {"title": "Result", "url": "https://real.com", "snippet": "snippet"}
                ],
            }
        )

        with patch("servers.base_tools.server._web_search", return_value=fake_result):
            result = engine.execute_tool("web_search", {"query": "Python testing"})

        data = json.loads(result)
        assert data["query"] == "Python testing"
        assert "results" in data
        assert len(data["results"]) > 0
        # No "note" key — mock implementation is gone
        assert "note" not in data

    @patch("core.chat.Client")
    def test_execute_tool_unknown(self, mock_client_class: Mock) -> None:
        """Test executing an unknown tool returns error."""
        engine = ChatEngine()
        result = engine.execute_tool("unknown_tool", {})

        # Standard Library
        import json

        data = json.loads(result)
        assert "error" in data
        assert "unknown_tool" in data["error"]

    @patch("core.chat.Client")
    def test_chat_success(
        self, mock_client_class: Mock, mock_ollama_client: Mock
    ) -> None:
        """Test successful chat interaction."""
        # Setup mock
        mock_client_class.return_value = mock_ollama_client

        engine = ChatEngine()
        response = engine.chat("Hello, brAIniac!")

        # Verify Ollama client was called
        mock_ollama_client.chat.assert_called_once()
        call_args = mock_ollama_client.chat.call_args

        # Check that messages were passed correctly
        assert call_args.kwargs["model"] == "llama3.1:8b-instruct-q4_K_M"
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2  # System message + user message
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello, brAIniac!"

        # Verify response
        assert response == "This is a test response from the mock LLM."

        # Verify memory was updated
        assert engine.memory.message_count() == 2  # User + assistant

    @patch("core.chat.Client")
    def test_chat_empty_response(self, mock_client_class: Mock) -> None:
        """Test handling of empty response from LLM."""
        # Setup mock with empty response
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": ""}}
        mock_client_class.return_value = mock_client

        engine = ChatEngine()
        response = engine.chat("Test message")

        # Should return fallback message for empty response
        assert "couldn't generate a response" in response

    @patch("core.chat.Client")
    def test_chat_ollama_error(self, mock_client_class: Mock) -> None:
        """Test handling of Ollama API errors."""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.chat.side_effect = Exception("Connection error")
        mock_client_class.return_value = mock_client

        engine = ChatEngine()
        response = engine.chat("Test message")

        # Should return error message
        assert "Error communicating with LLM" in response
        assert "Connection error" in response

    @patch("core.chat.Client")
    def test_clear_history(self, mock_client_class: Mock) -> None:
        """Test clearing conversation history."""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Response"},
        }
        mock_client_class.return_value = mock_client

        engine = ChatEngine()

        # Add some messages
        engine.chat("Message 1")
        engine.chat("Message 2")
        assert engine.memory.message_count() > 0

        # Clear history
        engine.clear_history()
        assert engine.memory.message_count() == 0

        # System message should still be present
        context = engine.memory.get_context()
        assert len(context) == 1
        assert context[0]["role"] == "system"

    @patch("core.chat.Client")
    def test_get_message_count(self, mock_client_class: Mock) -> None:
        """Test getting current message count."""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Response"},
        }
        mock_client_class.return_value = mock_client

        engine = ChatEngine()
        assert engine.get_message_count() == 0

        engine.chat("Message 1")
        assert engine.get_message_count() == 2  # User + assistant

        engine.chat("Message 2")
        assert engine.get_message_count() == 4

    @patch("core.chat.Client")
    def test_context_accumulation(
        self, mock_client_class: Mock, mock_ollama_client: Mock
    ) -> None:
        """Test that context accumulates across multiple chat calls."""
        mock_client_class.return_value = mock_ollama_client

        engine = ChatEngine()

        # First message
        engine.chat("First message")
        assert engine.memory.message_count() == 2

        # Second message
        engine.chat("Second message")
        assert engine.memory.message_count() == 4

        # Verify context contains all messages
        context = engine.memory.get_context()
        assert len(context) == 5  # System + 4 messages
        assert context[1]["content"] == "First message"
        assert context[3]["content"] == "Second message"

    @patch("core.chat.Client")
    def test_rolling_memory_integration(
        self, mock_client_class: Mock, mock_ollama_client: Mock
    ) -> None:
        """Test that rolling memory works correctly with chat engine."""
        mock_client_class.return_value = mock_ollama_client

        # Create engine with small rolling window
        engine = ChatEngine(max_context_messages=4)

        # Add 6 messages (3 exchanges, exceeds limit of 4)
        for i in range(3):
            engine.chat(f"Message {i + 1}")

        # Should only have last 4 messages (2 exchanges)
        assert engine.memory.message_count() == 4

        context = engine.memory.get_context()
        # System message + 4 rolling messages = 5 total
        assert len(context) == 5

    # -----------------------------------------------------------------------
    # Phase 2: memory_backend="disk"
    # -----------------------------------------------------------------------

    @patch("core.chat.Client")
    def test_memory_backend_disk(self, mock_client_class: Mock, tmp_path: Path) -> None:
        """ChatEngine uses DiskMemory when memory_backend='disk'."""
        p = tmp_path / "mem.jsonl"
        engine = ChatEngine(memory_backend="disk", persist_path=p)
        assert isinstance(engine.memory, DiskMemory)
        # System message must still be set
        ctx = engine.memory.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "system"

    @patch("core.chat.Client")
    def test_memory_backend_rolling_is_default(self, mock_client_class: Mock) -> None:
        """Default memory_backend keeps RollingMemory (no regression)."""
        engine = ChatEngine()
        assert isinstance(engine.memory, RollingMemory)

    # -----------------------------------------------------------------------
    # Phase 2: intent classifier gates research tools
    # -----------------------------------------------------------------------

    @patch("core.chat.Client")
    @patch("core.chat.needs_current_information", return_value=False)
    def test_intent_gates_tools_quiet_turn(
        self,
        mock_intent: Mock,
        mock_client_class: Mock,
        mock_ollama_client: Mock,
    ) -> None:
        """When classifier returns False, deep_research is NOT in turn schemas."""
        mock_client_class.return_value = mock_ollama_client
        engine = ChatEngine()

        engine.chat("What is the capital of France?")

        call_args = mock_ollama_client.chat.call_args
        tools_passed = call_args.kwargs.get("tools") or []
        tool_names = [t["function"]["name"] for t in tools_passed]
        assert "deep_research" not in tool_names
        # Base tools should still be present
        assert "get_current_time" in tool_names
        assert "web_search" in tool_names

    @patch("core.chat.Client")
    @patch("core.chat.needs_current_information", return_value=True)
    def test_intent_gates_tools_research_turn(
        self,
        mock_intent: Mock,
        mock_client_class: Mock,
        mock_ollama_client: Mock,
    ) -> None:
        """When classifier returns True, deep_research IS in turn schemas."""
        mock_client_class.return_value = mock_ollama_client
        engine = ChatEngine()

        engine.chat("What has NVIDIA released in the last 3 months?")

        call_args = mock_ollama_client.chat.call_args
        tools_passed = call_args.kwargs.get("tools") or []
        tool_names = [t["function"]["name"] for t in tools_passed]
        assert "deep_research" in tool_names

    @patch("core.chat.Client")
    def test_deep_research_registered_in_tool_schemas(
        self, mock_client_class: Mock
    ) -> None:
        """deep_research schema must be present in engine.tools after init."""
        engine = ChatEngine()
        assert "deep_research" in engine.tools
        schema = engine.tools["deep_research"]
        assert schema["function"]["name"] == "deep_research"
