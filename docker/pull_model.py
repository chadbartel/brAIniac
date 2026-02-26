"""docker/pull_model.py

One-shot model pull script run by the ``ollama-pull`` init container.

Streams the Ollama REST ``POST /api/pull`` endpoint and exits 0 on success,
1 on any error.  Uses only stdlib so no packages need to be installed.

Environment variables:
    OLLAMA_MODEL  — Full model tag to pull (e.g. ``llama3.1:8b-instruct-q4_K_M``).
    OLLAMA_HOST   — Ollama base URL (default: ``http://ollama:11434``).
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

_HOST: str = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
_MODEL: str = os.environ.get("OLLAMA_MODEL", "")

if not _MODEL:
    print("ERROR: OLLAMA_MODEL is not set.", flush=True)
    sys.exit(1)

print(f"Pulling '{_MODEL}' from {_HOST} ...", flush=True)

req = urllib.request.Request(
    f"{_HOST}/api/pull",
    data=json.dumps({"name": _MODEL}).encode(),
    headers={"Content-Type": "application/json"},
)

try:
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw_line in resp:
            line = raw_line.strip()
            if not line:
                continue
            try:
                d: dict = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "error" in d:
                print(f"ERROR: {d['error']}", flush=True)
                sys.exit(1)

            status: str = d.get("status", "")
            completed = d.get("completed")
            total = d.get("total")

            if total and completed is not None:
                pct = int(completed / total * 100)
                print(f"  {status} — {pct}%", flush=True)
            elif status:
                print(f"  {status}", flush=True)

except Exception as exc:  # noqa: BLE001
    print(f"ERROR: {exc}", flush=True)
    sys.exit(1)

print(f"'{_MODEL}' ready.", flush=True)
