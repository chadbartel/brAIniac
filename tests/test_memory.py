"""tests/test_memory.py

Unit tests for the RollingMemory class (core/memory.py).
Tests rolling context window, FIFO behavior, and system message preservation.
"""

from __future__ import annotations

# Third-Party Libraries
import pytest

# Local Modules
from core.memory import RollingMemory


class TestRollingMemory:
    """Test suite for RollingMemory class."""

    def test_initialization_default(self) -> None:
        """Test RollingMemory initializes with default parameters."""
        memory = RollingMemory()
        assert memory.max_messages == 20
        assert memory.message_count() == 0
        assert memory.get_context() == []

    def test_initialization_custom_max_messages(self) -> None:
        """Test RollingMemory initializes with custom max_messages."""
        memory = RollingMemory(max_messages=10)
        assert memory.max_messages == 10
        assert memory.message_count() == 0

    def test_set_system_message(self) -> None:
        """Test setting a system message."""
        memory = RollingMemory()
        system_prompt = "You are a helpful assistant."
        memory.set_system_message(system_prompt)

        context = memory.get_context()
        assert len(context) == 1
        assert context[0]["role"] == "system"
        assert context[0]["content"] == system_prompt

    def test_add_single_message(self) -> None:
        """Test adding a single message."""
        memory = RollingMemory()
        memory.add_message("user", "Hello!")

        assert memory.message_count() == 1
        context = memory.get_context()
        assert len(context) == 1
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "Hello!"

    def test_add_multiple_messages(self, sample_messages: list[dict[str, str]]) -> None:
        """Test adding multiple messages."""
        memory = RollingMemory()

        for msg in sample_messages:
            memory.add_message(msg["role"], msg["content"])

        assert memory.message_count() == len(sample_messages)
        context = memory.get_context()
        assert len(context) == len(sample_messages)

        # Verify order is preserved
        for i, msg in enumerate(sample_messages):
            assert context[i]["role"] == msg["role"]
            assert context[i]["content"] == msg["content"]

    def test_rolling_window_fifo_behavior(self) -> None:
        """Test that oldest messages are removed when max_messages is exceeded."""
        memory = RollingMemory(max_messages=3)

        # Add 5 messages (exceeds max_messages of 3)
        for i in range(5):
            memory.add_message("user", f"Message {i + 1}")

        # Should only have the last 3 messages
        assert memory.message_count() == 3
        context = memory.get_context()

        assert context[0]["content"] == "Message 3"
        assert context[1]["content"] == "Message 4"
        assert context[2]["content"] == "Message 5"

    def test_system_message_preserved_during_rolling(self) -> None:
        """Test that system message is preserved when messages roll off."""
        memory = RollingMemory(max_messages=2)
        memory.set_system_message("System prompt")

        # Add 5 messages (exceeds max_messages of 2)
        for i in range(5):
            memory.add_message("user", f"Message {i + 1}")

        context = memory.get_context()

        # System message should still be first
        assert len(context) == 3  # 1 system + 2 rolling messages
        assert context[0]["role"] == "system"
        assert context[0]["content"] == "System prompt"

        # Last 2 messages should be preserved
        assert context[1]["content"] == "Message 4"
        assert context[2]["content"] == "Message 5"

    def test_system_message_not_counted_toward_max(self) -> None:
        """Test that system message doesn't count toward max_messages limit."""
        memory = RollingMemory(max_messages=2)
        memory.set_system_message("System prompt")

        memory.add_message("user", "Message 1")
        memory.add_message("assistant", "Response 1")

        # message_count should be 2 (excluding system message)
        assert memory.message_count() == 2

        # But context should have 3 total (1 system + 2 messages)
        context = memory.get_context()
        assert len(context) == 3

    def test_clear_preserves_system_message(self) -> None:
        """Test that clear() removes messages but preserves system message."""
        memory = RollingMemory()
        memory.set_system_message("System prompt")

        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi!")

        assert memory.message_count() == 2

        memory.clear()

        # Messages cleared but system message preserved
        assert memory.message_count() == 0
        context = memory.get_context()
        assert len(context) == 1
        assert context[0]["role"] == "system"

    def test_update_system_message(self) -> None:
        """Test updating the system message."""
        memory = RollingMemory()
        memory.set_system_message("Original prompt")

        # Update system message
        memory.set_system_message("Updated prompt")

        context = memory.get_context()
        assert len(context) == 1
        assert context[0]["content"] == "Updated prompt"

    def test_empty_memory_returns_empty_context(self) -> None:
        """Test that empty memory returns empty context (no system message)."""
        memory = RollingMemory()
        assert memory.get_context() == []
        assert memory.message_count() == 0

    def test_alternating_roles(self) -> None:
        """Test adding messages with alternating user/assistant roles."""
        memory = RollingMemory()

        memory.add_message("user", "Question 1")
        memory.add_message("assistant", "Answer 1")
        memory.add_message("user", "Question 2")
        memory.add_message("assistant", "Answer 2")

        context = memory.get_context()
        assert len(context) == 4

        # Verify role alternation
        assert context[0]["role"] == "user"
        assert context[1]["role"] == "assistant"
        assert context[2]["role"] == "user"
        assert context[3]["role"] == "assistant"

    @pytest.mark.parametrize(
        "max_messages,num_messages,expected_count",
        [
            (5, 3, 3),  # Under limit
            (5, 5, 5),  # At limit
            (5, 10, 5),  # Over limit
            (1, 5, 1),  # Very small window
            (100, 10, 10),  # Very large window
        ],
    )
    def test_various_window_sizes(
        self, max_messages: int, num_messages: int, expected_count: int
    ) -> None:
        """Test rolling window behavior with various sizes."""
        memory = RollingMemory(max_messages=max_messages)

        for i in range(num_messages):
            memory.add_message("user", f"Message {i}")

        assert memory.message_count() == expected_count
