"""
brainiac/servers/system-server/src/system_server/api.py

FastAPI HTTP interface for the AutoGen orchestrator.

Endpoints:
  GET  /health          — liveness probe
  POST /run             — blocking execution, returns full result + messages
  POST /run/stream      — Server-Sent Events stream of agent messages + final answer
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .orchestrator import cfg, run_task

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("system-server.api")

# ---------------------------------------------------------------------------
# Thread pool for running the synchronous orchestrator
# ---------------------------------------------------------------------------
_executor = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="orchestrator"
)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="brAIniac System Server",
    version="0.1.0",
    description=(
        "AutoGen orchestrator API. Submit prompts and receive streamed "
        "agent thought processes and final answers."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user task prompt.")


class MessageEvent(BaseModel):
    agent: str
    recipient: str
    content: str
    type: str  # "message" | "function_call"


class RunResponse(BaseModel):
    answer: str
    messages: list[MessageEvent]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "server": "brainiac-system-server"}


@app.post("/run", response_model=RunResponse, tags=["orchestrator"])
async def run(body: RunRequest) -> RunResponse:
    """Execute a task and return the full answer plus all agent messages.

    This is a **blocking** call — it waits until the AutoGen group chat
    terminates.  For large tasks this can take several minutes.

    Args:
        body: JSON body with a ``prompt`` field.

    Returns:
        RunResponse with the final ``answer`` and the complete ``messages``
        list.
    """
    collected: list[MessageEvent] = []

    def _on_message(event: dict[str, Any]) -> None:
        collected.append(MessageEvent(**event))

    loop = asyncio.get_event_loop()
    answer: str = await loop.run_in_executor(
        _executor, lambda: run_task(body.prompt, on_message=_on_message)
    )
    return RunResponse(answer=answer, messages=collected)


@app.post("/run/stream", tags=["orchestrator"])
async def run_stream(body: RunRequest) -> StreamingResponse:
    """Stream agent messages as Server-Sent Events while the task runs.

    Each SSE event carries a JSON payload:

    - ``{"type": "message"|"function_call", "agent": "...", "recipient": "...",
      "content": "..."}``  — an agent turn or tool call
    - ``{"type": "answer", "content": "..."}``  — the final synthesised answer
    - ``{"type": "error",  "content": "..."}``  — if the orchestrator raised

    Clients should reconnect if the connection drops.

    Args:
        body: JSON body with a ``prompt`` field.

    Returns:
        StreamingResponse with ``text/event-stream`` content type.
    """
    msg_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _on_message(event: dict[str, Any]) -> None:
        msg_queue.put(event)

    def _run_in_thread() -> None:
        try:
            answer = run_task(body.prompt, on_message=_on_message)
            msg_queue.put({"type": "answer", "content": answer})
        except Exception as exc:
            logger.error("Streaming task error: %s", exc, exc_info=True)
            msg_queue.put({"type": "error", "content": str(exc)})
        finally:
            # Sentinel:告知生成器退出
            msg_queue.put(None)

    # Start orchestrator in a background thread
    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()

    async def _event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        while True:
            # Poll the queue without blocking the event loop
            event: dict[str, Any] | None = await loop.run_in_executor(
                None, msg_queue.get
            )
            if event is None:
                # Sentinel received — stream complete
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Entry point (called by the updated Poetry script)
# ---------------------------------------------------------------------------


def run_api() -> None:
    """Start the FastAPI server via uvicorn."""
    import os

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8300"))
    logger.info("Starting brainiac-system-server API on %s:%d", host, port)
    uvicorn.run(
        "system_server.api:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    run_api()
