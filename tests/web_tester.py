"""
tests/web_tester.py

Gradio web interface for the brAIniac system-server.

Features:
  - Chat window for submitting prompts and viewing final answers.
  - Expandable "Thought Process" accordion that streams every agent message
    and tool call as they arrive.

Usage:
    python web_tester.py [--url http://localhost:8300] [--port 7860]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Generator
from typing import Any

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# Defaults (overridden by CLI args)
# ---------------------------------------------------------------------------
DEFAULT_SERVER_URL = "http://localhost:8300"

# ---------------------------------------------------------------------------
# Agent colour map (Gradio Markdown styling via emoji prefix)
# ---------------------------------------------------------------------------
AGENT_ICONS: dict[str, str] = {
    "OrchestratorAgent": "🧠",
    "ResearchAgent": "🔍",
    "HumanProxy": "👤",
    "GroupChatManager": "📋",
}
DEFAULT_ICON = "🤖"


def _icon(agent: str) -> str:
    return AGENT_ICONS.get(agent, DEFAULT_ICON)


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------


def _stream_events(
    server_url: str,
    prompt: str,
    history: list[dict[str, str]] | None = None,
) -> Generator[tuple[str, str, str], None, None]:
    """Connect to /run/stream and yield (chat_message, thought_log, status).

    Yields tuples of:
        - chat_message : cumulative chat markdown (answer placeholder → answer)
        - thought_log  : cumulative thought-process markdown
        - status       : short status string

    Args:
        server_url: Base URL of the system-server.
        prompt: The current user prompt.
        history: Prior conversation turns to provide context to the
            orchestrator. Each turn is a dict with ``role`` and ``content``.
    """
    url = f"{server_url.rstrip('/')}/run/stream"
    thought_lines: list[str] = []
    answer_text = "_Thinking…_"

    yield answer_text, "_Connecting to orchestrator…_", "connecting"

    try:
        payload: dict[str, Any] = {"prompt": prompt}
        if history:
            payload["history"] = history

        with httpx.Client(timeout=None) as client:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()

                yield answer_text, "_Orchestration started._\n\n---\n", "running"

                for raw_line in resp.iter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("event:"):
                        break  # stream finished
                    if not line.startswith("data:"):
                        continue

                    payload = line[len("data:") :].strip()
                    try:
                        event: dict[str, Any] = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "message")
                    content = event.get("content", "")
                    agent = event.get("agent", "?")
                    recipient = event.get("recipient", "?")

                    if etype == "answer":
                        answer_text = content
                        thought_lines.append("\n---\n**✅ Task complete.**")
                        yield (
                            answer_text,
                            "\n".join(thought_lines),
                            "complete",
                        )
                        return

                    if etype == "error":
                        thought_lines.append(
                            f"\n> ❌ **Error:** {_escape_md(content)}"
                        )
                        yield (
                            f"_Error: {content}_",
                            "\n".join(thought_lines),
                            "error",
                        )
                        return

                    if etype == "function_call":
                        # Parse "[tool: search_web] {...}"
                        parts = content.split("]", 1)
                        tool_name = (
                            parts[0].lstrip("[tool: ").strip()
                            if len(parts) == 2
                            else content
                        )
                        tool_args = parts[1].strip() if len(parts) == 2 else ""
                        thought_lines.append(
                            f"\n> ⚙️ **Tool call** `{tool_name}` "
                            f"by **{agent}**"
                            + (
                                f"\n> ```json\n> {tool_args[:400]}\n> ```"
                                if tool_args
                                else ""
                            )
                        )
                    else:
                        icon = _icon(agent)
                        escaped = _escape_md(content[:1000]) + (
                            "…" if len(content) > 1000 else ""
                        )
                        thought_lines.append(
                            f"\n**{icon} {agent}** → **{recipient}**\n\n"
                            f"{escaped}\n"
                        )

                    yield answer_text, "\n".join(thought_lines), "running"

    except httpx.ConnectError:
        msg = (
            f"Cannot connect to system-server at **{server_url}**.\n\n"
            "Make sure the Docker stack is running:\n"
            "```bash\ncd docker && docker compose up -d\n```"
        )
        yield msg, msg, "error"
    except httpx.HTTPStatusError as exc:
        msg = f"HTTP error {exc.response.status_code}: {exc}"
        yield msg, msg, "error"


def _escape_md(text: str) -> str:
    """Minimal Markdown escape for user-generated content.

    Args:
        text: Raw string that may contain Markdown-special characters.

    Returns:
        Escaped string safe to embed inside Markdown paragraphs.
    """
    for ch in (
        "\\",
        "`",
        "*",
        "_",
        "{",
        "}",
        "[",
        "]",
        "(",
        ")",
        "#",
        "+",
        "-",
        ".",
        "!",
    ):
        text = text.replace(ch, f"\\{ch}")
    return text


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def _health_status(server_url: str) -> str:
    """Return a short health status string.

    Args:
        server_url: Base URL to probe.
    """
    try:
        resp = httpx.get(f"{server_url.rstrip('/')}/health", timeout=4)
        if resp.status_code == 200:
            return f"✅ Connected to {server_url}"
        return f"⚠️ Server responded with HTTP {resp.status_code}"
    except Exception as exc:
        return f"❌ Cannot reach {server_url} — {exc}"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------


def build_interface(server_url: str) -> gr.Blocks:
    """Construct and return the Gradio Blocks application.

    Args:
        server_url: Base URL of the system-server to target.

    Returns:
        A configured ``gr.Blocks`` instance (not yet launched).
    """

    def submit(
        user_message: str,
        history: list[dict[str, str]],
    ) -> Generator[
        tuple[list[dict[str, str]], str, str],
        None,
        None,
    ]:
        """Handle chat submission and stream updates.

        Args:
            user_message: The typed user prompt.
            history: Existing chat history in Gradio messages format.

        Yields:
            Tuples of (updated_history, thought_log, status).
        """
        if not user_message.strip():
            yield history, "", "idle"
            return

        # Capture prior turns BEFORE appending the new exchange.
        # Strip to only role+content — Gradio 6 injects extra keys (e.g.
        # `metadata`) with non-string values that fail the server's
        # dict[str, str] validation and produce a 422.
        prior_history: list[dict[str, str]] = [
            {"role": t["role"], "content": t.get("content") or ""}
            for t in history
            if t.get("role") and t.get("content")
        ]

        # Append user turn then an empty assistant placeholder
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": ""},
        ]

        for answer, thought_log, status in _stream_events(
            server_url, user_message, history=prior_history if prior_history else None
        ):
            # Update the last assistant turn in place
            history[-1] = {"role": "assistant", "content": answer}
            yield history, thought_log, status

    def check_health() -> str:
        return _health_status(server_url)

    # theme/css moved to launch() per Gradio 6.0 migration
    with gr.Blocks(title="brAIniac") as demo:
        gr.Markdown(
            "# 🧠 brAIniac\n"
            "> Local-first AI orchestrator · AutoGen + FastMCP + Ollama"
        )

        with gr.Row():
            health_box = gr.Markdown(value=_health_status(server_url))
            refresh_btn = gr.Button("↺ Refresh", scale=0, size="sm")

        refresh_btn.click(fn=check_health, outputs=health_box)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="brAIniac Chat",
                    height=520,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask brAIniac anything…",
                        label="",
                        scale=5,
                        submit_btn=True,
                        autofocus=True,
                    )
                    clear_btn = gr.Button("🗑 Clear", scale=0, size="sm")

            with gr.Column(scale=2):
                status_badge = gr.Markdown("**Status:** idle")
                with gr.Accordion("🔍 Thought Process", open=False):
                    thought_box = gr.Markdown(
                        value="_Agent reasoning will appear here during execution._",
                        elem_classes=["thought-box"],
                    )

        # Wire events
        msg_input.submit(
            fn=submit,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, thought_box, status_badge],
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )

        clear_btn.click(
            fn=lambda: ([], "_Cleared._", "idle"),
            outputs=[chatbot, thought_box, status_badge],
        )

        gr.Markdown(
            "<small>Targeting system-server at "
            f"<code>{server_url}</code></small>",
            visible=True,
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args and launch the Gradio app."""
    parser = argparse.ArgumentParser(
        prog="web_tester",
        description="brAIniac Gradio web tester",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_SERVER_URL,
        help=f"System-server base URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port for the Gradio app (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    args = parser.parse_args()

    demo = build_interface(server_url=args.url)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="emerald",
            neutral_hue="slate",
        ),
        css=(
            ".thought-box textarea { font-family: monospace; font-size: 0.82em; }\n"
            "footer { display: none !important; }"
        ),
    )


if __name__ == "__main__":
    main()
