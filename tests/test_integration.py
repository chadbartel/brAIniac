"""tests/test_integration.py

Integration tests for brAIniac components working together.
Tests end-to-end workflows and component interactions.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from core.chat import ChatEngine
from core.memory import RollingMemory


class TestIntegration:
    """Integration tests for brAIniac system."""

    @patch("core.chat.Client")
    def test_memory_chat_integration(self, mock_client_class: Mock) -> None:
        """Test that ChatEngine and RollingMemory work together correctly."""
        # Setup mock
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Integration test response"},
        }
        mock_client_class.return_value = mock_client

        engine = ChatEngine(max_context_messages=3)

        # Send multiple messages
        engine.chat("Message 1")
        engine.chat("Message 2")
        engine.chat("Message 3")

        # Memory should be at limit
        assert engine.get_message_count() == 3

        # Add one more message - should trigger rolling
        engine.chat("Message 4")

        # Still at limit but oldest message should be gone
        assert engine.get_message_count() == 3

        # Verify correct messages are preserved
        context = engine.memory.get_context()
        messages_content = [msg["content"] for msg in context if msg["role"] != "system"]

        # Should have messages 2, 3, 4 and their responses
        # (Message 1 should have rolled off)
        assert "Message 1" not in str(messages_content)
        assert "Message 4" in str(messages_content)

    @patch("core.chat.Client")
    def test_tool_execution_in_context(self, mock_client_class: Mock) -> None:
        """Test that tool execution results can be incorporated into context."""
        mock_client = Mock()
        mock_client.chat.return_value = {
            "message": {"content": "Tool-based response"},
        }
        mock_client_class.return_value = mock_client

        engine = ChatEngine()

        # Execute a tool
        time_result = engine.execute_tool("get_current_time", {})
        assert time_result is not None

        # Chat should still work after tool execution
        response = engine.chat("What time is it?")
        assert response == "Tool-based response"

    @patch("core.chat.Client")
    def test_full_conversation_flow(self, mock_client_class: Mock) -> None:
        """Test a complete conversation flow."""
        mock_client = Mock()
        responses = [
            {"message": {"content": "Hello! How can I help you?"}},
            {"message": {"content": "The weather is nice today."}},
            {"message": {"content": "Goodbye! Have a great day!"}},
        ]
        mock_client.chat.side_effect = responses
        mock_client_class.return_value = mock_client

        engine = ChatEngine()

        # Conversation flow
        r1 = engine.chat("Hi there!")
        assert "Hello" in r1

        r2 = engine.chat("What's the weather?")
        assert "weather" in r2

        r3 = engine.chat("Thanks, bye!")
        assert "Goodbye" in r3

        # Verify context accumulation
        assert engine.get_message_count() == 6  # 3 user + 3 assistant

    @pytest.mark.slow
    def test_memory_independence(self) -> None:
        """Test that multiple memory instances are independent."""
        memory1 = RollingMemory(max_messages=5)
        memory2 = RollingMemory(max_messages=5)

        memory1.add_message("user", "Message for memory 1")
        memory2.add_message("user", "Message for memory 2")

        context1 = memory1.get_context()
        context2 = memory2.get_context()

        # Each should have only its own message
        assert len(context1) == 1
        assert len(context2) == 1
        assert context1[0]["content"] != context2[0]["content"]

    @patch("core.chat.Client")
    def test_error_recovery_in_conversation(self, mock_client_class: Mock) -> None:
        """Test that conversation can continue after an error."""
        mock_client = Mock()

        # First call succeeds, second fails, third succeeds
        mock_client.chat.side_effect = [
            {"message": {"content": "Success 1"}},
            Exception("Temporary error"),
            {"message": {"content": "Success 2"}},
        ]
        mock_client_class.return_value = mock_client

        engine = ChatEngine()

        # First message succeeds
        r1 = engine.chat("Message 1")
        assert "Success 1" in r1

        # Second message fails
        r2 = engine.chat("Message 2")
        assert "Error" in r2

        # Third message succeeds
        r3 = engine.chat("Message 3")
        assert "Success 2" in r3

        # All messages should be in memory
        assert engine.get_message_count() == 6  # 3 user + 3 assistant (including error)

    @patch("core.chat.Client")
    def test_clear_and_resume(self, mock_client_class: Mock) -> None:
        """Test clearing history and resuming conversation."""
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": "Response"}}
        mock_client_class.return_value = mock_client

        engine = ChatEngine()

        # Have a conversation
        engine.chat("Before clear 1")
        engine.chat("Before clear 2")
        assert engine.get_message_count() == 4

        # Clear history
        engine.clear_history()
        assert engine.get_message_count() == 0

        # Resume conversation
        engine.chat("After clear")
        assert engine.get_message_count() == 2

        # System message should still be present
        context = engine.memory.get_context()
        assert context[0]["role"] == "system"

    @patch("core.chat.Client")
    def test_large_context_handling(self, mock_client_class: Mock) -> None:
        """Test handling of large conversations with rolling memory."""
        mock_client = Mock()
        mock_client.chat.return_value = {"message": {"content": "OK"}}
        mock_client_class.return_value = mock_client

        engine = ChatEngine(max_context_messages=10)

        # Send 20 messages (exceeds rolling window)
        for i in range(20):
            engine.chat(f"Message {i + 1}")

        # Should only have last 10 messages
        assert engine.get_message_count() == 10

        # Context size should be stable
        context = engine.memory.get_context()
        assert len(context) == 11  # System + 10 messages
