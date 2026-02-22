"""
brainiac/servers/system-server/src/system_server/intent_classifier.py

Lightweight structural validator for OrchestratorAgent output.

This module is intentionally domain-agnostic. It does NOT make routing
decisions — the LLM owns that. It only validates that whatever the LLM
produced contains a recognised mode token, and exposes helpers for
extracting the CLARIFY question text if present.

Adding a new mode to the system:
    1. Add the token name to ``Mode`` below.
    2. Add handling in ``orchestrator.py`` (no changes needed here).
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum
from typing import Final

logger = logging.getLogger("system-server.intent")

# Matches the first occurrence of a mode token anywhere in the first 300
# characters of LLM output — robust to small-model format drift where the
# token ends up mid-sentence rather than on the first line.
_MODE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(CHAT|RESEARCH|CLARIFY)\b",
    re.IGNORECASE,
)

# Captures the question text after "CLARIFY:" up to TERMINATE or end-of-string.
_CLARIFY_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"CLARIFY:\s*(.+?)(?:\s*TERMINATE|$)",
    re.IGNORECASE | re.DOTALL,
)


class Mode(StrEnum):
    """Valid orchestration modes emitted by the OrchestratorAgent."""

    CHAT = "CHAT"
    RESEARCH = "RESEARCH"
    CLARIFY = "CLARIFY"
    # Sentinel — model produced unrecognised output. Caller decides fallback.
    UNKNOWN = "UNKNOWN"


def extract_mode(llm_output: str) -> Mode:
    """Parse the mode token from raw LLM output.

    Scans the first 300 characters so minor format drift (token buried
    mid-sentence rather than on the first line) does not cause a miss.

    Args:
        llm_output: Raw text from the OrchestratorAgent's first response.

    Returns:
        The recognised :class:`Mode`, or ``Mode.UNKNOWN`` if none found.
    """
    snippet = llm_output[:300]
    match = _MODE_PATTERN.search(snippet)
    if not match:
        logger.warning(
            "No valid mode token in orchestrator output (first 300 chars): %r",
            snippet,
        )
        return Mode.UNKNOWN

    mode = Mode(match.group(1).upper())
    logger.info("Intent classifier resolved mode: %s", mode)
    return mode


def extract_clarify_question(llm_output: str) -> str | None:
    """Extract the clarifying question text from a CLARIFY-mode response.

    Args:
        llm_output: Raw LLM output expected to contain ``CLARIFY: <question>``.

    Returns:
        The question string with surrounding whitespace stripped, or ``None``
        if the pattern is not found.
    """
    match = _CLARIFY_PATTERN.search(llm_output)
    if not match:
        logger.warning(
            "CLARIFY mode detected but no question text found in: %r",
            llm_output[:200],
        )
        return None
    return match.group(1).strip()
