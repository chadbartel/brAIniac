@echo off
:: run_harness.bat
::
:: Launches the brAIniac live browser test harness.
:: Usage:
::   run_harness.bat                          (defaults: localhost:11434, env model)
::   run_harness.bat llama3.2:3b              (override model)
::   run_harness.bat llama3.2:3b 7862         (override model + port)
::
:: Environment variables (optional):
::   OLLAMA_BASE_URL  — Ollama host URL  (default: http://localhost:11434)
::   OLLAMA_MODEL     — Model name       (default: llama3.2:3b)
::   HARNESS_PORT     — Gradio port      (default: 7861)

setlocal

:: ── Apply optional positional overrides ─────────────────────────────────────
if not "%~1"=="" set OLLAMA_MODEL=%~1
if not "%~2"=="" set HARNESS_PORT=%~2

:: ── Defaults ─────────────────────────────────────────────────────────────────
if not defined OLLAMA_BASE_URL set OLLAMA_BASE_URL=http://localhost:11434
if not defined OLLAMA_MODEL    set OLLAMA_MODEL=llama3.2:3b
if not defined HARNESS_PORT    set HARNESS_PORT=7861

echo.
echo  ============================================================
echo   brAIniac - Live Test Harness
echo  ============================================================
echo   Ollama host : %OLLAMA_BASE_URL%
echo   Model       : %OLLAMA_MODEL%
echo   Port        : %HARNESS_PORT%
echo  ============================================================
echo.

:: ── Confirm Poetry is available ──────────────────────────────────────────────
where poetry >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: 'poetry' not found on PATH.
    echo  Install Poetry from: https://python-poetry.org/docs/
    pause
    exit /b 1
)

:: ── Run the harness ──────────────────────────────────────────────────────────
poetry run python tests/live_harness.py

if %errorlevel% neq 0 (
    echo.
    echo  Harness exited with code %errorlevel%.
    pause
)

endlocal
