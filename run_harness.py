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

Recognised environment variables:
    OLLAMA_BASE_URL   — Ollama host URL       (default: http://localhost:11434)
    OLLAMA_MODEL      — Model name            (default: llama3.1:8b-instruct-q4_K_M)
    HARNESS_PORT      — Gradio listen port    (default: 7861)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``core`` and ``tests`` are
# importable when this script is run from any working directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# live_harness calls load_dotenv() internally before resolving its constants,
# so .env is always loaded before the Gradio UI is constructed.
from tests.live_harness import main  # noqa: E402

if __name__ == "__main__":
    main()
