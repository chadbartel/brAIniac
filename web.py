"""web.py

Gradio web interface for brAIniac.
Exposes the ChatEngine over HTTP at 0.0.0.0:7860 (mapped to host port 80).

This module is the entrypoint for the ``brainiac-web`` Docker service.
It is a single-user, local-first interface — the ChatEngine is a
module-level singleton that holds the rolling conversation context.

Exposed interfaces:
    demo (gr.Blocks): The Gradio application.  Launch via ``python web.py``.
"""

from __future__ import annotations

# Standard Library
import os
import logging

# Third-Party Libraries
import gradio as gr
from dotenv import load_dotenv

# Local Modules
from core.chat import ChatEngine

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment — same vars as CLI / docker-compose)
# ---------------------------------------------------------------------------
_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
_ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_max_messages: int = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))

# ---------------------------------------------------------------------------
# Singleton engine — safe for local single-user deployment
# ---------------------------------------------------------------------------
engine: ChatEngine = ChatEngine(
    model=_model,
    ollama_host=_ollama_host,
    max_context_messages=_max_messages,
)
logger.info("ChatEngine initialised: model=%s  host=%s", _model, _ollama_host)


# ---------------------------------------------------------------------------
# Gradio handler functions
# ---------------------------------------------------------------------------


def respond(
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    """Send a user message and append both turns to the Gradio chat history.

    The ChatEngine maintains its own rolling context internally; the returned
    ``history`` list is used only for Gradio display.

    Args:
        message: The user's input text.
        history: Current Gradio chat history (list of role/content dicts).

    Returns:
        A tuple of (cleared input text, updated chat history).
    """
    if not message.strip():
        return "", history

    reply: str = engine.chat(message)
    updated: list[dict[str, str]] = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return "", updated


def clear_history() -> list[dict[str, str]]:
    """Reset the ChatEngine conversation context and clear the display.

    Returns:
        Empty history list consumed by the Gradio Chatbot component.
    """
    engine.clear_history()
    logger.info("Conversation history cleared via web UI")
    return []


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="brAIniac") as demo:
    gr.Markdown(
        "# 🧠 brAIniac\n"
        "*Local-first, uncensored AI — running entirely on your hardware.*"
    )

    chatbot = gr.Chatbot(
        label="brAIniac",
        height=540,
        layout="bubble",
        buttons=["copy"],
    )

    with gr.Row():
        txt = gr.Textbox(
            placeholder="Type your message and press Enter…",
            show_label=False,
            container=False,
            scale=9,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑️  Clear history", variant="secondary")
        gr.Markdown(f"**Model:** `{_model}` &nbsp;|&nbsp; **Ollama:** `{_ollama_host}`")

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------
    txt.submit(respond, inputs=[txt, chatbot], outputs=[txt, chatbot])
    send_btn.click(respond, inputs=[txt, chatbot], outputs=[txt, chatbot])
    clear_btn.click(clear_history, outputs=chatbot)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
