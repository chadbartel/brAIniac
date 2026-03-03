"""tests/test_memory.py

Unit tests for the RollingMemory and DiskMemory classes (core/memory.py).
"""

from __future__ import annotations

# Standard Library
from pathlib import Path

# Third-Party Libraries
import pytest

# Local Modules
from core.memory import DiskMemory, RollingMemory


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


# ---------------------------------------------------------------------------
# DiskMemory tests
# ---------------------------------------------------------------------------


class TestDiskMemory:
    """Test suite for the Phase 2 DiskMemory backend."""

    # ------------------------------------------------------------------
    # Basic interface compliance (mirrors TestRollingMemory tests)
    # ------------------------------------------------------------------

    def test_initialization_creates_file(self, tmp_path: Path) -> None:
        """DiskMemory creates the parent directory and JSONL file on first write."""
        p = tmp_path / "sub" / "mem.jsonl"
        mem = DiskMemory(max_messages=10, persist_path=p)
        mem.add_message("user", "hello")
        assert p.exists()

    def test_set_system_message(self, tmp_path: Path) -> None:
        """System message is stored and returned at the head of context."""
        p = tmp_path / "mem.jsonl"
        mem = DiskMemory(persist_path=p)
        mem.set_system_message("You are a test bot.")
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "system"
        assert ctx[0]["content"] == "You are a test bot."

    def test_add_message_and_count(self, tmp_path: Path) -> None:
        """message_count reflects the total number of non-system messages."""
        p = tmp_path / "mem.jsonl"
        mem = DiskMemory(persist_path=p)
        mem.add_message("user", "ping")
        mem.add_message("assistant", "pong")
        assert mem.message_count() == 2

    def test_get_context_includes_system_and_messages(self, tmp_path: Path) -> None:
        """get_context returns system message first, then conversation messages."""
        p = tmp_path / "mem.jsonl"
        mem = DiskMemory(persist_path=p)
        mem.set_system_message("sys")
        mem.add_message("user", "hello")
        mem.add_message("assistant", "hi")
        ctx = mem.get_context()
        assert ctx[0]["role"] == "system"
        assert ctx[1]["role"] == "user"
        assert ctx[2]["role"] == "assistant"

    def test_clear_removes_messages_preserves_system(self, tmp_path: Path) -> None:
        """clear() wipes messages but keeps the system prompt."""
        p = tmp_path / "mem.jsonl"
        mem = DiskMemory(persist_path=p)
        mem.set_system_message("sys")
        mem.add_message("user", "Message 1")
        mem.add_message("assistant", "Response 1")
        mem.clear()
        assert mem.message_count() == 0
        ctx = mem.get_context()
        assert len(ctx) == 1
        assert ctx[0]["role"] == "system"

    # ------------------------------------------------------------------
    # FIFO context window with full history
    # ------------------------------------------------------------------

    def test_context_window_is_bounded(self, tmp_path: Path) -> None:
        """get_context() returns at most max_messages entries (FIFO)."""
        p = tmp_path / "mem.jsonl"
        mem = DiskMemory(max_messages=3, persist_path=p)
        for i in range(7):
            mem.add_message("user", f"Message {i}")
        ctx = mem.get_context()
        # Only last 3 messages in context
        assert len(ctx) == 3
        # But full history is stored
        assert mem.message_count() == 7

    def test_context_fifo_order(self, tmp_path: Path) -> None:
        """get_context() returns the most recent messages when windowed."""
        p = tmp_path / "mem.jsonl"
        mem = DiskMemory(max_messages=2, persist_path=p)
        for i in range(5):
            mem.add_message("user", f"Message {i}")
        ctx = mem.get_context()
        assert ctx[0]["content"] == "Message 3"
        assert ctx[1]["content"] == "Message 4"

    def test_message_count_returns_full_history_not_window(
        self, tmp_path: Path
    ) -> None:
        """message_count() is the total stored count, not the windowed slice."""
        p = tmp_path / "mem.jsonl"
        mem = DiskMemory(max_messages=2, persist_path=p)
        for i in range(6):
            mem.add_message("user", f"msg {i}")
        assert mem.message_count() == 6

    # ------------------------------------------------------------------
    # Persistence across re-instantiation
    # ------------------------------------------------------------------

    def test_persistence_across_restart(self, tmp_path: Path) -> None:
        """Messages written in one instance are available in a new instance."""
        p = tmp_path / "mem.jsonl"

        mem1 = DiskMemory(persist_path=p)
        mem1.set_system_message("persisted sys")
        mem1.add_message("user", "first session msg")

        # Simulate restart
        mem2 = DiskMemory(persist_path=p)
        assert mem2.message_count() == 1
        ctx = mem2.get_context()
        assert ctx[0]["role"] == "system"
        assert ctx[0]["content"] == "persisted sys"
        assert ctx[1]["content"] == "first session msg"

    def test_multiple_restarts_accumulate_messages(self, tmp_path: Path) -> None:
        """Each instantiation appends; total count grows correctly."""
        p = tmp_path / "mem.jsonl"

        for i in range(3):
            m = DiskMemory(persist_path=p)
            m.add_message("user", f"session {i}")

        final = DiskMemory(persist_path=p)
        assert final.message_count() == 3

    # ------------------------------------------------------------------
    # Clear persists to disk
    # ------------------------------------------------------------------

    def test_clear_reflected_on_disk(self, tmp_path: Path) -> None:
        """After clear(), a new instance sees zero messages."""
        p = tmp_path / "mem.jsonl"
        mem1 = DiskMemory(persist_path=p)
        mem1.set_system_message("sys")
        mem1.add_message("user", "should be erased")
        mem1.clear()

        mem2 = DiskMemory(persist_path=p)
        assert mem2.message_count() == 0
        ctx = mem2.get_context()
        # System message survives the clear
        assert len(ctx) == 1
        assert ctx[0]["role"] == "system"
