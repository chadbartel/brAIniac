"""tests/live_harness.py

Live browser-based test harness for brAIniac.
Connects to a real local Ollama instance and exposes an interactive
Gradio chat UI with streaming responses, model switching, and diagnostics.

Configuration is loaded from the project-root ``.env`` file via
``python-dotenv``.  Recognised environment variables:

- ``OLLAMA_BASE_URL``   — Ollama host (default: ``http://localhost:11434``)
- ``OLLAMA_MODEL``      — Model name  (default: ``llama3.1:8b-instruct-q4_K_M``)
- ``HARNESS_PORT``      — Gradio port (default: ``7861``)

Run with:
    python run_harness.py          # recommended — loads .env automatically
    # or directly:
    python tests/live_harness.py
"""

from __future__ import annotations

# Standard Library
import os
import sys
import json
import time
import logging
import urllib.request
from datetime import datetime
from typing import Any, Generator

# Third-Party Libraries
import gradio as gr
from ddgs import DDGS
from dotenv import load_dotenv
from gradio.themes import Soft

# Local Modules
from core.chat import ChatEngine

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (Ollama / OpenAI function-calling schema)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current system date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for factual, real-time, or current information. "
                "ONLY call this for questions that require up-to-date facts such as "
                "current weather, news, prices, sports scores, general knowledge, "
                "math, coding, or recent events. Do NOT call this for greetings, "
                "casual conversation, opinions, or creative writing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def _execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a named tool and return its JSON-encoded result.

    Uses in-process DDG for web_search (per architecture §4.4) so the
    harness never depends on the research-server SSE proxy being alive.

    Args:
        name: Tool function name (``get_current_time`` or ``web_search``).
        arguments: Parsed arguments dict from the model’s tool call.

    Returns:
        JSON-encoded result string, or an error dict on failure.
    """
    logger.info("[tool] executing %s with args %s", name, arguments)

    if name == "get_current_time":
        now = datetime.now()
        return json.dumps(
            {
                "iso_format": now.isoformat(),
                "readable": now.strftime("%A, %B %d, %Y at %I:%M:%S %p"),
                "timezone": "Local system time",
            }
        )

    if name == "web_search":
        query: str = str(arguments.get("query", ""))
        # Guard against llama3.2:3b passing the parameter schema dict as the
        # value instead of a plain integer.
        max_results_raw = arguments.get("max_results", 5)
        if isinstance(max_results_raw, dict):
            max_results_raw = max_results_raw.get("max_results", 5)
        max_results: int = int(max_results_raw)
        try:
            with DDGS() as ddgs:
                hits = [
                    {"title": r["title"], "url": r["href"], "snippet": r["body"]}
                    for r in ddgs.text(query, max_results=max_results)
                ]
            return json.dumps(
                {"query": query, "results_count": len(hits), "results": hits},
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.error("[web_search] DDG failure: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc)})

    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _str_content(content: None | str | list[dict[str, Any]]) -> str:
    """Normalise a Gradio / Ag2 message content value to a plain string.

    Gradio's ``type="messages"`` chatbot can return history entries whose
    ``content`` field is a multi-part list such as::

        [{'text': 'hello', 'type': 'text'}]

    rather than a bare string.  Passing such a list directly to the Ollama
    client causes a Pydantic ``string_type`` validation error.  This helper
    normalises all three possible shapes to a plain ``str``.

    Args:
        content: Raw content from a Gradio or Ag2 message dict.

    Returns:
        Plain string representation of the content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Multi-part content list — concatenate all text parts.
    return " ".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Load .env before reading any os.environ values so that the project-root
# .env file is always the authoritative configuration source.
load_dotenv()

DEFAULT_OLLAMA_HOST: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
DEFAULT_MAX_CTX: int = 20
HARNESS_PORT: int = int(os.environ.get("HARNESS_PORT", "7861"))

# ---------------------------------------------------------------------------
# Ollama probing helpers
# ---------------------------------------------------------------------------


def _probe_ollama(host: str, timeout: int = 3) -> bool:
    """Return True if the Ollama HTTP API is reachable at *host*.

    Args:
        host: Base URL of the Ollama instance (e.g. ``http://localhost:11434``).
        timeout: Connection timeout in seconds.

    Returns:
        ``True`` when the ``/api/tags`` endpoint responds with HTTP 200.
    """
    try:
        url = f"{host.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _list_models(host: str) -> list[str]:
    """Return a list of locally available Ollama model names.

    Args:
        host: Base URL of the Ollama instance.

    Returns:
        Sorted list of model name strings, or an empty list on failure.
    """
    try:
        url = f"{host.rstrip('/')}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data: dict[str, Any] = json.loads(resp.read())
        return sorted(m["name"] for m in data.get("models", []))
    except Exception as exc:
        logger.warning("Could not fetch model list: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Session state — one engine per browser session
# ---------------------------------------------------------------------------


class HarnessSession:
    """Holds the active ChatEngine and related state for one browser session.

    Attributes:
        engine: The live (or mock-fallback) ChatEngine instance.
        ollama_live: Whether we successfully connected to a real Ollama server.
        host: The Ollama host URL currently in use.
        model: The model name currently in use.
    """

    def __init__(
        self, host: str = DEFAULT_OLLAMA_HOST, model: str = DEFAULT_MODEL
    ) -> None:
        """Initialise the harness session.

        Args:
            host: Ollama base URL.
            model: Ollama model name.
        """
        self.host: str = host
        self.model: str = model
        self.ollama_live: bool = _probe_ollama(host)
        self.engine: ChatEngine = self._build_engine()

    def _build_engine(self) -> ChatEngine:
        """Construct a ChatEngine, falling back gracefully if Ollama is offline.

        Returns:
            A configured ``ChatEngine`` instance.
        """
        engine = ChatEngine(
            model=self.model,
            ollama_host=self.host,
            max_context_messages=DEFAULT_MAX_CTX,
        )
        if not self.ollama_live:
            logger.warning(
                "Ollama not reachable at %s — responses will show an error banner.",
                self.host,
            )
        return engine

    def reconnect(self, host: str, model: str) -> str:
        """Reconnect to (possibly different) Ollama host / model.

        Args:
            host: New Ollama base URL.
            model: New model name.

        Returns:
            Human-readable status string.
        """
        self.host = host.strip() or DEFAULT_OLLAMA_HOST
        self.model = model.strip() or DEFAULT_MODEL
        self.ollama_live = _probe_ollama(self.host)
        self.engine = self._build_engine()

        if self.ollama_live:
            return f"✅ Connected to **{self.host}** — model `{self.model}`"
        return (
            f"⚠️ Could not reach Ollama at **{self.host}**. "
            "Check that `ollama serve` is running."
        )

    def status_badge(self) -> str:
        """Return a short markdown status badge.

        Returns:
            One-line markdown string indicating live / offline status.
        """
        if self.ollama_live:
            return f"🟢 **Live** | `{self.model}` @ `{self.host}`"
        return f"🔴 **Offline** | Ollama not reachable at `{self.host}`"


# Single global session (shared across all browser tabs at localhost)
_session: HarnessSession = HarnessSession()


# ---------------------------------------------------------------------------
# Chat handler — streaming generator for Gradio
# ---------------------------------------------------------------------------


def chat_stream(
    message: str,
    history: list[dict[str, str]],
) -> Generator[str, None, None]:
    """Stream a response from the live ChatEngine for Gradio's chatbot.

    This function is used as the ``fn`` argument to ``gr.ChatInterface`` with
    ``type="messages"``.  Gradio calls it as a generator; each ``yield``
    appends characters to the in-progress assistant bubble in real time.

    Args:
        message: The user's latest message.
        history: The full chat history in ``{"role": ..., "content": ...}``
            format (managed by Gradio).

    Yields:
        Accumulated response text, growing character by character until the
        full reply is assembled.
    """
    if not message.strip():
        yield ""
        return

    if not _session.ollama_live:
        # Re-probe — user may have started Ollama after launching the harness.
        _session.ollama_live = _probe_ollama(_session.host)
        if not _session.ollama_live:
            yield (
                "⚠️ **Ollama is not running.**\n\n"
                "Start it with `ollama serve` then click **Reconnect** in the "
                "Settings tab, or restart the harness."
            )
            return

    t0 = time.perf_counter()

    try:
        # Rebuild memory from Gradio history so the engine stays in sync.
        # Normalise content through _str_content: Gradio's type="messages"
        # chatbot can return content as a list[dict] for multimodal turns,
        # which the Ollama Pydantic model rejects as not a valid string.
        _session.engine.memory.clear()
        for turn in history:
            role = turn.get("role", "")
            content = _str_content(turn.get("content", ""))
            if role in ("user", "assistant") and content:
                _session.engine.memory.add_message(role, content)

        # ------------------------------------------------------------------
        # Pass 1 (non-streaming): let the model declare which tools it needs.
        # We send the bare user message plus tool schemas so the model can
        # emit structured tool-call JSON cleanly without stream fragmentation.
        # ------------------------------------------------------------------
        _session.engine.memory.add_message("user", message)
        context: list[dict[str, Any]] = _session.engine.memory.get_context()

        first_resp = _session.engine.client.chat(
            model=_session.model,
            messages=context,
            tools=TOOLS,
        )
        first_msg = first_resp.message
        tool_calls = first_msg.tool_calls or []

        if tool_calls:
            # ------------------------------------------------------------------
            # Pre-injection pattern (architecture §4.4): small quantized models
            # do not reliably act on role="tool" response messages — they see
            # the data but revert to training-data answers anyway.
            #
            # Instead, execute every requested tool and embed all results as
            # plain-text context directly inside the user turn.  The model then
            # receives a single, self-contained prompt it cannot ignore.
            # ------------------------------------------------------------------
            injected_parts: list[str] = []
            for tc in tool_calls:
                result_json = _execute_tool(
                    tc.function.name, dict(tc.function.arguments)
                )
                injected_parts.append(
                    f"[{tc.function.name} result]\n{result_json}"
                )

            injected_context = "\n\n".join(injected_parts)
            augmented_user_msg = (
                f"<tool_results>\n{injected_context}\n</tool_results>\n\n"
                f"Using ONLY the tool results above, answer the following "
                f"question. Do not say you lack access to real-time data — "
                f"the data has already been retrieved for you.\n\n"
                f"Question: {message}"
            )

            # Replace the last user message in context with the augmented one,
            # then stream a single synthesis call — no role="tool" messages.
            synthesis_context = context[:-1] + [
                {"role": "user", "content": augmented_user_msg}
            ]

            accumulated = ""
            stream = _session.engine.client.chat(
                model=_session.model,
                messages=synthesis_context,
                stream=True,
            )
            for chunk in stream:
                delta: str = (
                    chunk.get("message", {}).get("content", "")
                    if isinstance(chunk, dict)
                    else getattr(getattr(chunk, "message", None), "content", "") or ""
                )
                accumulated += delta
                yield accumulated

        else:
            # No tools needed — yield the Pass 1 response directly.
            accumulated = _str_content(first_msg.content)
            yield accumulated

        elapsed = time.perf_counter() - t0
        yield accumulated + f"\n\n<sub>\u23f1 {elapsed:.2f}s</sub>"

        # Persist the final assistant message in engine memory.
        _session.engine.memory.add_message("assistant", accumulated)

    except Exception as exc:
        logger.error("Stream error: %s", exc, exc_info=True)
        yield f"❌ **Error:** `{exc}`\n\nCheck that Ollama is running and the model is loaded."


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def apply_settings(host: str, model: str) -> tuple[str, gr.Dropdown]:
    """Apply new connection settings and refresh the model dropdown.

    Args:
        host: Ollama base URL entered by the user.
        model: Model name entered by the user.

    Returns:
        Tuple of (status markdown, updated Dropdown component).
    """
    status = _session.reconnect(host, model)
    models = _list_models(_session.host)
    dropdown_update = gr.Dropdown(
        choices=models or [model],
        value=_session.model,
        label="Available models",
    )
    return status, dropdown_update


def get_connection_status() -> str:
    """Return the current connection status badge.

    Returns:
        Markdown status string.
    """
    _session.ollama_live = _probe_ollama(_session.host)
    return _session.status_badge()


def clear_history_fn() -> list[dict[str, str]]:
    """Clear the chat history and engine memory.

    Returns:
        Empty history list.
    """
    _session.engine.clear_history()
    return []


def get_diagnostics() -> str:
    """Return a markdown diagnostic report for the current session.

    Returns:
        Multi-section markdown string.
    """
    live = _probe_ollama(_session.host)
    models = _list_models(_session.host) if live else []
    msg_count = _session.engine.get_message_count()
    max_msgs = _session.engine.memory.max_messages
    utilization = (msg_count / max_msgs * 100) if max_msgs > 0 else 0.0

    model_list_md = (
        "\n".join(f"  - `{m}`" for m in models)
        if models
        else "  *(unavailable — Ollama offline)*"
    )

    return f"""
### 🔍 Live Diagnostics

| Field | Value |
|---|---|
| Ollama host | `{_session.host}` |
| Reachable | {"✅ Yes" if live else "❌ No"} |
| Active model | `{_session.model}` |
| Context messages | {msg_count} / {max_msgs} ({utilization:.0f}%) |
| Python | `{sys.version.split()[0]}` |
| Gradio | `{gr.__version__}` |

---

### 🗂 Available Models

{model_list_md}

---

### 💡 Tips

- **Slow first response?** Ollama loads the model on the first request — subsequent turns are faster.
- **Model not listed?** Run `ollama pull <model-name>` then click **Refresh / Reconnect**.
- **Context window full?** Click **Clear Chat** to reset — the rolling window will keep the last {max_msgs} turns automatically.
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface.

    Returns:
        A fully configured ``gr.Blocks`` instance ready to ``launch()``.
    """
    initial_models = _list_models(_session.host)
    initial_status = _session.status_badge()

    with gr.Blocks(
        title="brAIniac — Live Test Harness",
        theme=Soft(
            primary_hue="violet",
            secondary_hue="slate",
        ),
    ) as ui:
        gr.Markdown(
            f"""
# 🧠 brAIniac — Live Test Harness

Interactive real-time chat connected to your local Ollama instance.

{initial_status}
""",
        )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # Tab 1: Live Chat
            # ------------------------------------------------------------------
            with gr.Tab("💬 Live Chat"):
                status_bar = gr.Markdown(initial_status)

                chat_iface = gr.ChatInterface(
                    fn=chat_stream,
                    title="",
                    description=None,
                    chatbot=gr.Chatbot(
                        label="brAIniac",
                        height=520,
                        show_label=False,
                        render_markdown=True,
                        layout="bubble",
                    ),
                    textbox=gr.Textbox(
                        placeholder="Ask anything — press Enter or click Submit…",
                        show_label=False,
                        autofocus=True,
                    ),
                    submit_btn="Send ↵",
                    stop_btn="⏹ Stop",
                )

                # Refresh the status badge after each chat turn.
                chat_iface.chatbot.change(
                    fn=get_connection_status,
                    outputs=[status_bar],
                )

            # ------------------------------------------------------------------
            # Tab 2: Settings / Connection
            # ------------------------------------------------------------------
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### Ollama Connection")

                with gr.Row():
                    host_input = gr.Textbox(
                        label="Ollama host",
                        value=_session.host,
                        placeholder="http://localhost:11434",
                        scale=3,
                    )
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=initial_models or [DEFAULT_MODEL],
                        value=_session.model,
                        allow_custom_value=True,
                        scale=3,
                    )
                    reconnect_btn = gr.Button(
                        "🔌 Reconnect", variant="primary", scale=1
                    )

                settings_status = gr.Markdown()

                reconnect_btn.click(
                    fn=apply_settings,
                    inputs=[host_input, model_dropdown],
                    outputs=[settings_status, model_dropdown],
                )

                gr.Markdown("""
---
### Context Window

The rolling memory window keeps the last **{n}** conversation turns.
When the window is full the oldest messages are silently dropped so the
model never exceeds its context limit.

To reset mid-conversation use the **🗑 Clear** button in the chat tab.
""".format(n=DEFAULT_MAX_CTX))

            # ------------------------------------------------------------------
            # Tab 3: Diagnostics
            # ------------------------------------------------------------------
            with gr.Tab("🩺 Diagnostics"):
                diag_btn = gr.Button("🔄 Refresh Diagnostics", variant="secondary")
                diag_output = gr.Markdown(get_diagnostics())

                diag_btn.click(fn=get_diagnostics, outputs=[diag_output])

    return ui


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the live test harness."""
    print("=" * 60)
    print("  brAIniac — Live Test Harness")
    print("=" * 60)
    print(f"  Ollama host : {_session.host}")
    print(f"  Model       : {_session.model}")
    print(
        f"  Status      : {'🟢 Connected' if _session.ollama_live else '🔴 Offline (start ollama serve)'}"
    )
    print(f"  Harness URL : http://127.0.0.1:{HARNESS_PORT}")
    print("=" * 60)
    print()

    ui = build_ui()
    ui.launch(
        server_name="127.0.0.1",
        server_port=HARNESS_PORT,
        share=False,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
