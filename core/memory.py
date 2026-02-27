"""core/memory.py

Rolling context window implementation to prevent context exhaustion.
Maintains the last N messages to keep VRAM usage under control.
"""

from __future__ import annotations

# Standard Library
from typing import Any


class RollingMemory:
    """Rolling context window that stores the last N messages.

    This prevents context exhaustion on 8GB VRAM by maintaining a fixed-size
    conversation history. In later phases, this will be replaced with Letta
    (MemGPT) for OS-level virtual context management.
    """

    def __init__(self, max_messages: int = 20) -> None:
        """Initialize rolling memory with a fixed capacity.

        Args:
            max_messages: Maximum number of messages to retain (default 20).
                         Includes system messages, user messages, and assistant
                         responses. A typical value of 20 allows ~10 back-and-forth
                         exchanges before the oldest messages roll off.
        """
        self.max_messages = max_messages
        self._messages: list[dict[str, Any]] = []
        self._system_message: dict[str, str] | None = None

    def set_system_message(self, content: str) -> None:
        """Set or update the system message.

        The system message is always preserved and prepended to the context,
        and does not count toward the max_messages limit.

        Args:
            content: The system prompt content.
        """
        self._system_message = {"role": "system", "content": content}

    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the rolling window.

        Args:
            role: Message role - 'user', 'assistant', or 'system'.
            content: The message content.
        """
        message: dict[str, str] = {"role": role, "content": content}
        self._messages.append(message)

        # Enforce the rolling window - keep only the last max_messages
        if len(self._messages) > self.max_messages:
            # Remove oldest message (FIFO)
            self._messages.pop(0)

    def get_context(self) -> list[dict[str, Any]]:
        """Retrieve the current conversation context.

        Returns:
            List of message dictionaries ready for Ollama API consumption.
            System message (if set) is always first, followed by rolling history.
        """
        context: list[dict[str, Any]] = []

        # Always include system message first if it exists
        if self._system_message:
            context.append(self._system_message)

        # Append the rolling message history
        context.extend(self._messages)

        return context

    def clear(self) -> None:
        """Clear all messages from the rolling window.

        The system message is preserved.
        """
        self._messages.clear()

    def message_count(self) -> int:
        """Get the current number of messages in the rolling window.

        Returns:
            Number of messages (excludes system message).
        """
        return len(self._messages)
