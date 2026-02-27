#!/usr/bin/env python3
"""run_harness.py

Python entry point for the brAIniac live test harness.

Replaces run_harness.sh / run_harness.bat.  Configuration is read from the
project-root ``.env`` file automatically — no shell variable exports required.

Usage:
    python run_harness.py

Override any value by editing ``.env`` or setting the variable in your shell
before running:
    OLLAMA_MODEL=llama3.2:3b python run_harness.py

Recognised environment variables (all read from .env):
    OLLAMA_HOST       — Ollama host URL       (default: http://localhost:11434)
    OLLAMA_MODEL      — Model name            (default: llama3.1:8b-instruct-q4_K_M)
    HARNESS_PORT      — Gradio listen port    (default: 7861)
"""

from __future__ import annotations

# Standard Library
import os
import sys
import json
import urllib.request
from pathlib import Path

# Third-Party Libraries
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env first so all subsequent os.environ reads pick up project config.
load_dotenv(_ROOT / ".env")

_HOST: str = (
    os.environ.get("OLLAMA_HOST")
    or os.environ.get("OLLAMA_BASE_URL")
    or "http://localhost:11434"
)
_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")


# ---------------------------------------------------------------------------
# Pre-flight helpers
# ---------------------------------------------------------------------------


def _check_ollama_running(host: str) -> bool:
    """Return True if Ollama is reachable, exit with a message if not.

    Args:
        host: Base URL of the Ollama instance.

    Returns:
        ``True`` when reachable.
    """
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5):
            return True
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"  ✗  Cannot reach Ollama at {host}")
        print(f"     Error: {exc}")
        print()
        print("  Ensure Ollama is running:")
        print("    ollama serve          (local install)")
        print("    docker compose -f docker/docker-compose.yml up -d ollama   (Docker)")
        print()
        sys.exit(1)


def _list_local_models(host: str) -> list[str]:
    """Return names of models already pulled into the local Ollama instance.

    Args:
        host: Base URL of the Ollama instance.

    Returns:
        Sorted list of model name strings.
    """
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=5) as resp:
            data: dict = json.loads(resp.read())
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:  # noqa: BLE001
        return []


def _pull_model(host: str, model: str) -> None:
    """Stream-pull *model* from the Ollama registry with a progress indicator.

    Args:
        host: Base URL of the Ollama instance.
        model: Full model tag to pull (e.g. ``llama3.2:3b``).
    """
    # Third-Party Libraries
    import ollama as _ollama  # local import — not needed if model already present

    client = _ollama.Client(host=host)
    print(f"  ↓  Pulling '{model}' — this may take a few minutes...")
    last_status: str = ""
    try:
        for progress in client.pull(model, stream=True):
            status: str = getattr(progress, "status", "") or ""
            completed = getattr(progress, "completed", None)
            total = getattr(progress, "total", None)
            if total and completed is not None:
                pct = int(completed / total * 100)
                line = f"  ↓  {status} — {pct}%"
            else:
                line = f"  ↓  {status}"
            if line != last_status:
                print(f"\r{line:<60}", end="", flush=True)
                last_status = line
        print()  # newline after final progress line
        print(f"  ✓  '{model}' ready.")
    except Exception as exc:  # noqa: BLE001
        print()
        print(f"  ✗  Pull failed: {exc}")
        print("     Check your internet connection and that the model tag is correct.")
        sys.exit(1)


def _ensure_model(host: str, model: str) -> None:
    """Pull *model* if it is not already present in the local Ollama instance.

    Args:
        host: Base URL of the Ollama instance.
        model: Full model tag to ensure is available.
    """
    available = _list_local_models(host)
    # Ollama may append a qualifier (e.g. ":latest") — check prefix too.
    already_present = any(
        m == model or m.startswith(f"{model}:") or model.startswith(f"{m}:")
        for m in available
    )
    if already_present:
        print(f"  ✓  Model '{model}' already available.")
    else:
        _pull_model(host, model)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run pre-flight checks then launch the Gradio harness."""
    print("=" * 60)
    print("  brAIniac — Pre-flight checks")
    print("=" * 60)
    print(f"  Host  : {_HOST}")
    print(f"  Model : {_MODEL}")
    print()

    _check_ollama_running(_HOST)
    _ensure_model(_HOST, _MODEL)

    print()

    # Delegate to the live harness — .env is already loaded above so
    # live_harness.load_dotenv() is effectively a no-op (already set).
    # Local Modules
    from tests.live_harness import main as _harness_main  # noqa: PLC0415

    _harness_main()


if __name__ == "__main__":
    main()
