#!/usr/bin/env bash
# run_harness.sh
#
# Launches the brAIniac live browser test harness.
#
# Usage:
#   ./run_harness.sh                           (defaults: localhost:11434, env model)
#   ./run_harness.sh llama3.2:3b               (override model)
#   ./run_harness.sh llama3.2:3b 7862          (override model + port)
#
# Environment variables (optional):
#   OLLAMA_BASE_URL  — Ollama host URL  (default: http://localhost:11434)
#   OLLAMA_MODEL     — Model name       (default: llama3.1:8b-instruct-q4_K_M)
#   HARNESS_PORT     — Gradio port      (default: 7861)

set -euo pipefail

# ── Apply optional positional overrides ─────────────────────────────────────
[[ -n "${1:-}" ]] && export OLLAMA_MODEL="$1"
[[ -n "${2:-}" ]] && export HARNESS_PORT="$2"

# ── Defaults ─────────────────────────────────────────────────────────────────
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b-instruct-q4_K_M}"
export HARNESS_PORT="${HARNESS_PORT:-7861}"

echo
echo " ============================================================"
echo "  brAIniac - Live Test Harness"
echo " ============================================================"
echo "  Ollama host : ${OLLAMA_BASE_URL}"
echo "  Model       : ${OLLAMA_MODEL}"
echo "  Port        : ${HARNESS_PORT}"
echo " ============================================================"
echo

# ── Confirm Poetry is available ──────────────────────────────────────────────
if ! command -v poetry &>/dev/null; then
    echo " ERROR: 'poetry' not found on PATH."
    echo " Install Poetry from: https://python-poetry.org/docs/"
    exit 1
fi

# ── Run the harness ──────────────────────────────────────────────────────────
poetry run python tests/live_harness.py
