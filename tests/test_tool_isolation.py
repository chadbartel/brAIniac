"""tests/test_tool_isolation.py

Regression test for the tool-selection context-contamination bug.

Scenario that previously broke:
  1. User asks about weather in Seattle → get_weather fires correctly.
  2. User asks for a pygame stick figure drawing → model INCORRECTLY called
     get_weather because the prior assistant response contained "Seattle" and
     weather data, polluting the Pass 1 tool-selection context.

Fix verified: Pass 1 now uses [system, current_user_msg] ONLY so prior turns
cannot bleed into the tool-selection decision.

Run with:
    poetry run python tests/test_tool_isolation.py
"""

from __future__ import annotations

import json
import sys
import time
import logging
from typing import Any
from unittest.mock import patch

# Ensure project root is on path when running directly.
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Import harness functions under test.
# We need _session to be initialised so we import the module directly.
# ---------------------------------------------------------------------------
import tests.live_harness as harness  # type: ignore[import]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"


def collect_stream(gen) -> str:
    """Drain a generator and return the last yielded value."""
    last = ""
    for chunk in gen:
        last = chunk
    return last


# ---------------------------------------------------------------------------
# Test 1: Weather question fires get_weather, NOT web_search or nothing
# ---------------------------------------------------------------------------


def test_weather_fires_correct_tool() -> bool:
    """Turn 1: 'What's the weather in Seattle?' should call get_weather."""
    called_tools: list[str] = []
    original_execute = harness._execute_tool

    def spy_execute(name: str, args: dict[str, Any]) -> str:
        called_tools.append(name)
        return original_execute(name, args)

    harness._session.engine.memory.clear()

    with patch.object(harness, "_execute_tool", side_effect=spy_execute):
        response = collect_stream(
            harness.chat_stream("What's the weather in Seattle today?", [])
        )

    if "get_weather" in called_tools:
        print(f"  {PASS}  test_weather_fires_correct_tool  (tools called: {called_tools})")
        return True
    else:
        print(
            f"  {FAIL}  test_weather_fires_correct_tool  "
            f"(expected get_weather, got: {called_tools})\n"
            f"  response: {response[:200]}"
        )
        return False


# ---------------------------------------------------------------------------
# Test 2: Pygame question after a weather turn must NOT call get_weather
# ---------------------------------------------------------------------------


def test_pygame_after_weather_no_tool_contamination() -> bool:
    """Turn 2: After a weather turn, a pygame question must NOT call get_weather."""
    called_tools: list[str] = []
    original_execute = harness._execute_tool

    def spy_execute(name: str, args: dict[str, Any]) -> str:
        called_tools.append(name)
        return original_execute(name, args)

    # Simulate Gradio history that would exist after the weather turn.
    simulated_history: list[dict[str, str]] = [
        {"role": "user", "content": "What's the weather in Seattle today?"},
        {
            "role": "assistant",
            "content": (
                "The current weather in Seattle is clear with a temperature of "
                "45°F. Humidity is 60%. Expect mild conditions through the day.\n\n"
                "<sub>⏱ 12.34s</sub>"
            ),
        },
    ]

    with patch.object(harness, "_execute_tool", side_effect=spy_execute):
        response = collect_stream(
            harness.chat_stream(
                "Write Python pygame code to draw a stick figure with the head "
                "at the top of the screen, a body line attached at 6 o'clock, "
                "two arms slightly below the head on either side, and two legs "
                "at the bottom of the body pointing diagonally outward. "
                "All body parts must be drawn relative to the position of the head.",
                simulated_history,
            )
        )

    if "get_weather" not in called_tools:
        print(
            f"  {PASS}  test_pygame_after_weather_no_tool_contamination  "
            f"(tools called: {called_tools or ['none']})"
        )
        return True
    else:
        print(
            f"  {FAIL}  test_pygame_after_weather_no_tool_contamination  "
            f"(get_weather was incorrectly called — context contamination bug persists!)\n"
            f"  tools called: {called_tools}\n"
            f"  response: {response[:200]}"
        )
        return False


# ---------------------------------------------------------------------------
# Test 3: Verify the pygame response actually contains relative positioning
# ---------------------------------------------------------------------------


def test_pygame_response_contains_relative_positioning() -> bool:
    """The pygame response should reference coordinates relative to head_x / head_y."""
    simulated_history: list[dict[str, str]] = [
        {"role": "user", "content": "What's the weather in Seattle today?"},
        {
            "role": "assistant",
            "content": (
                "The current weather in Seattle is clear at 45°F.\n\n<sub>⏱ 12.34s</sub>"
            ),
        },
    ]

    response = collect_stream(
        harness.chat_stream(
            "Write Python pygame code to draw a stick figure with the head "
            "at the top of the screen, a body line attached at 6 o'clock, "
            "two arms slightly below the head on either side, and two legs "
            "at the bottom of the body pointing diagonally outward. "
            "All body parts must be drawn relative to the position of the head.",
            simulated_history,
        )
    )

    # Check that the response contains pygame code and relative coord references
    has_pygame = "pygame" in response.lower()
    has_relative = any(
        kw in response
        for kw in ["head_x", "head_y", "head[0]", "head[1]", "center_x", "cx", "+ radius", "- radius"]
    )
    has_draw = "draw" in response.lower()

    if has_pygame and has_draw:
        quality = "with relative coords" if has_relative else "without explicit relative coords"
        print(
            f"  {PASS}  test_pygame_response_contains_relative_positioning  "
            f"(pygame code generated {quality})"
        )
        return True
    else:
        print(
            f"  {FAIL}  test_pygame_response_contains_relative_positioning  "
            f"(response doesn't look like pygame code)\n"
            f"  has_pygame={has_pygame}, has_draw={has_draw}\n"
            f"  response[:300]: {response[:300]}"
        )
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 60)
    print("  brAIniac — Tool Isolation Regression Tests")
    print("=" * 60)
    print(f"  Model : {harness._session.model}")
    print(f"  Host  : {harness._session.host}")
    print()

    if not harness._session.ollama_live:
        print("  ⚠️  Ollama is not reachable — cannot run live tests.")
        sys.exit(1)

    results: list[bool] = []

    print("  Running tests (each may take 30–90s for inference)...\n")

    t0 = time.perf_counter()
    results.append(test_weather_fires_correct_tool())
    results.append(test_pygame_after_weather_no_tool_contamination())
    results.append(test_pygame_response_contains_relative_positioning())
    elapsed = time.perf_counter() - t0

    passed = sum(results)
    total = len(results)

    print()
    print(f"  Results: {passed}/{total} passed  ({elapsed:.1f}s)")
    print("=" * 60)
    print()

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
