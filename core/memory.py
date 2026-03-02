"""core/memory.py

Memory backend implementations for brAIniac.

Provides:
  - BaseMemory  — abstract interface all backends must satisfy
  - RollingMemory — Phase 1 in-process rolling window (unchanged)
  - DiskMemory  — Phase 2 JSONL-backed persistent memory; acts as the
                  Letta bridge until Phase 3 replaces it with the real
                  MemGPT agent.
"""

from __future__ import annotations

# Standard Library
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseMemory(ABC):
    """Abstract base class that every memory backend must implement.

    All backends share the same interface so ChatEngine can swap them
    without touching its own logic.
    """

    @abstractmethod
    def set_system_message(self, content: str) -> None:
        """Set or replace the persistent system prompt."""
        ...

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """Append a new user/assistant message."""
        ...

    @abstractmethod
    def get_context(self) -> list[dict[str, Any]]:
        """Return the full context list ready for the Ollama API."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Flush conversation history (system message must survive)."""
        ...

    @abstractmethod
    def message_count(self) -> int:
        """Return the number of non-system messages currently stored."""
        ...


# ---------------------------------------------------------------------------
# Phase 1 backend — unchanged rolling window
# ---------------------------------------------------------------------------


class RollingMemory(BaseMemory):
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


# ---------------------------------------------------------------------------
# Phase 2 backend — JSONL-backed persistent memory
# ---------------------------------------------------------------------------


class DiskMemory(BaseMemory):
    """Persistent JSONL-backed memory — the Phase 2 Letta bridge.

    Writes every message to a JSONL file so that conversation history
    survives process restarts.  The in-memory ``_messages`` list holds
    *all* history; ``get_context()`` returns only the last
    ``max_messages`` entries so the rolling context window is preserved.

    Migration path: replace this class with a real Letta / MemGPT agent
    in Phase 3 by substituting a ``LettaMemory(BaseMemory)`` shim.
    """

    def __init__(
        self,
        max_messages: int = 20,
        persist_path: Path | None = None,
    ) -> None:
        """Initialise DiskMemory.

        Args:
            max_messages: Maximum messages returned by ``get_context()``
                (does **not** limit how many are persisted to disk).
            persist_path: Path to the JSONL history file.  Defaults to
                the ``MEMORY_PERSIST_PATH`` env var, then
                ``~/.brainiac/memory.jsonl``.
        """
        self.max_messages = max_messages
        self._system_message: dict[str, str] | None = None
        self._messages: list[dict[str, Any]] = []

        if persist_path is not None:
            self._path = persist_path
        else:
            env_path = os.getenv("MEMORY_PERSIST_PATH")
            self._path = (
                Path(env_path)
                if env_path
                else Path.home() / ".brainiac" / "memory.jsonl"
            )

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Hydrate from disk on startup
        self._load_from_disk()
        logger.info(
            "DiskMemory initialised: path=%s, loaded=%d messages",
            self._path,
            len(self._messages),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """Read all messages from the JSONL file into ``_messages``."""
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry: dict[str, Any] = json.loads(line)
                    if entry.get("role") == "system":
                        self._system_message = {
                            "role": "system",
                            "content": entry["content"],
                        }
                    else:
                        self._messages.append(entry)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "DiskMemory: could not load history from %s: %s", self._path, exc
            )

    def _append_to_disk(self, entry: dict[str, Any]) -> None:
        """Append a single JSON line to the JSONL file atomically."""
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error(
                "DiskMemory: could not write to %s: %s", self._path, exc, exc_info=True
            )

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    def set_system_message(self, content: str) -> None:
        """Set or replace the system prompt.

        The system message is **not** written to the JSONL append log so
        that re-loading the file doesn't accumulate duplicates.  Instead,
        the file is rewritten in-place only when the system message
        changes (rare).

        Args:
            content: The new system prompt text.
        """
        self._system_message = {"role": "system", "content": content}
        # Rewrite the file header so persistence reflects the new prompt.
        self._rewrite_file()

    def _rewrite_file(self) -> None:
        """Rewrite the entire JSONL file (used after system-message changes)."""
        try:
            with self._path.open("w", encoding="utf-8") as fh:
                if self._system_message:
                    fh.write(
                        json.dumps(self._system_message, ensure_ascii=False) + "\n"
                    )
                for msg in self._messages:
                    fh.write(json.dumps(msg, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error(
                "DiskMemory: could not rewrite %s: %s", self._path, exc, exc_info=True
            )

    def add_message(self, role: str, content: str) -> None:
        """Append a message to in-memory history and flush to disk.

        Args:
            role: ``'user'`` or ``'assistant'``.
            content: Message text.
        """
        entry: dict[str, str] = {"role": role, "content": content}
        self._messages.append(entry)
        self._append_to_disk(entry)

    def get_context(self) -> list[dict[str, Any]]:
        """Return the bounded context window ready for the Ollama API.

        Always prepends the system message when set.  Only the last
        ``max_messages`` conversation turns are included so context stays
        within the configured limit.

        Returns:
            List of message dicts (system + up to ``max_messages`` turns).
        """
        context: list[dict[str, Any]] = []
        if self._system_message:
            context.append(self._system_message)
        context.extend(self._messages[-self.max_messages :])
        return context

    def clear(self) -> None:
        """Clear conversation history but preserve the system message.

        Both ``_messages`` and the JSONL file are truncated.
        """
        self._messages.clear()
        self._rewrite_file()
        logger.info("DiskMemory cleared: %s", self._path)

    def message_count(self) -> int:
        """Return the total number of stored messages (full history, not truncated).

        Returns:
            ``len(self._messages)`` — includes all history, not just the
            context window slice.
        """
        return len(self._messages)
