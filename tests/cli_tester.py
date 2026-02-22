"""
tests/cli_tester.py

Interactive terminal client for the brAIniac system-server.

Streams agent thought processes, tool calls, and final answers in real-time
with rich colour-coded formatting.

Usage:
    python cli_tester.py [--url http://localhost:8300]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Colour scheme — each pipeline stage gets a distinct colour
# ---------------------------------------------------------------------------
AGENT_COLOURS: dict[str, str] = {
    # Layer 1 — tool router (small model)
    "ToolRouter": "bold yellow",
    # Tool execution stage (individual tool names also use this)
    "search_web": "bold magenta",
    "store_memory": "bold magenta",
    "query_memory": "bold magenta",
    # Layer 2 — generator (large model)
    "Generator": "bold green",
    # Final synthesised answer emitted under the legacy agent name
    "OrchestratorAgent": "bold cyan",
}
TOOL_COLOUR = "bold magenta"
ANSWER_COLOUR = "bold white on dark_green"
ERROR_COLOUR = "bold white on red"
DEFAULT_AGENT_COLOUR = "bold blue"

CUSTOM_THEME = Theme(
    {
        "orchestrator": "bold cyan",
        "researcher": "bold green",
        "human": "bold yellow",
        "tool": "bold magenta",
        "answer": "bold white",
        "error": "bold red",
        "dim": "dim white",
    }
)

console = Console(theme=CUSTOM_THEME)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent_colour(agent_name: str) -> str:
    return AGENT_COLOURS.get(agent_name, DEFAULT_AGENT_COLOUR)


def _render_event(event: dict[str, Any]) -> None:
    """Render a single SSE event to the console.

    Args:
        event: Parsed JSON event dict from the SSE stream.
    """
    etype = event.get("type", "message")
    content = event.get("content", "")
    agent = event.get("agent", "?")
    recipient = event.get("recipient", "?")

    if etype == "answer":
        console.print()
        console.print(
            Panel(
                escape(content),
                title="[bold]brAIniac — Final Answer[/bold]",
                border_style="green",
                padding=(1, 2),
            )
        )
        return

    if etype == "error":
        console.print(
            Panel(
                f"[{ERROR_COLOUR}]{escape(content)}[/{ERROR_COLOUR}]",
                title="[bold red]Error[/bold red]",
                border_style="red",
            )
        )
        return

    colour = _agent_colour(agent)

    if etype == "function_call":
        parts = content.split("]", 1)
        tool_tag = parts[0].lstrip("[") if len(parts) == 2 else content
        tool_args = parts[1].strip() if len(parts) == 2 else ""
        label = Text()
        label.append(f"  ⚙ {tool_tag}", style=TOOL_COLOUR)
        if tool_args:
            label.append(f"  {tool_args[:120]}", style="dim")
        console.print(label)
        return

    # Regular message
    header = Text()
    header.append(f"[{agent}]", style=colour)
    header.append(" → ", style="dim")
    header.append(f"[{recipient}]", style=_agent_colour(recipient))

    # Truncate very long content for readability; full text streamed anyway
    body = escape(content[:2000]) + ("…" if len(content) > 2000 else "")
    console.print(header)
    console.print(f"  [dim]{body}[/dim]")


# ---------------------------------------------------------------------------
# Streaming interaction
# ---------------------------------------------------------------------------


def stream_prompt(
    base_url: str,
    prompt: str,
    router_model: str | None = None,
    generator_model: str | None = None,
) -> None:
    """Send a prompt to the system-server and stream the response.

    Args:
        base_url: Base URL of the system-server (e.g. http://localhost:8300).
        prompt: User task prompt.
        router_model: Optional Layer 1 router model override.
        generator_model: Optional Layer 2 generator model override.
    """
    url = f"{base_url.rstrip('/')}/run/stream"
    console.print(Rule("[dim]Starting orchestration[/dim]", style="cyan"))

    payload: dict[str, Any] = {"prompt": prompt}
    if router_model:
        payload["router_model"] = router_model
    if generator_model:
        payload["generator_model"] = generator_model

    with httpx.Client(timeout=None) as client:
        try:
            with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("event:"):
                        # "event: done" — stream finished
                        break
                    if line.startswith("data:"):
                        payload = line[len("data:") :].strip()
                        try:
                            event = json.loads(payload)
                            _render_event(event)
                        except json.JSONDecodeError:
                            console.print(
                                f"[dim](unparseable SSE data: {payload[:80]})[/dim]"
                            )
        except httpx.ConnectError:
            console.print(
                f"[bold red]Cannot connect to system-server at {base_url}.[/bold red]\n"
                "Is the Docker stack running?  "
                "Try: [bold]cd docker && docker compose up -d[/bold]"
            )
            sys.exit(1)
        except httpx.HTTPStatusError as exc:
            console.print(
                f"[bold red]HTTP {exc.response.status_code}:[/bold red] {exc}"
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def check_health(base_url: str) -> bool:
    """Return True if the system-server responds to the health probe.

    Args:
        base_url: Base URL of the system-server.
    """
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def main() -> None:
    """Interactive CLI loop."""
    parser = argparse.ArgumentParser(
        prog="cli_tester",
        description="brAIniac interactive CLI tester",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8300",
        help="Base URL of the system-server (default: http://localhost:8300)",
    )
    parser.add_argument(
        "--router-model",
        default=None,
        metavar="MODEL",
        help="Override the Layer 1 router model (e.g. 'llama3.2:3b').",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        metavar="MODEL",
        help="Override the Layer 2 generator model (e.g. 'dolphin-llama3').",
    )
    args = parser.parse_args()
    base_url: str = args.url

    model_info: list[str] = []
    if args.router_model:
        model_info.append(f"Router: {args.router_model}")
    if args.generator_model:
        model_info.append(f"Generator: {args.generator_model}")
    model_line = ("  " + "  |  ".join(model_info)) if model_info else ""

    console.print(
        Panel(
            "[bold cyan]brAIniac CLI Tester[/bold cyan]\n"
            f"[dim]Connecting to: {base_url}[/dim]"
            + (f"\n[dim]{model_line}[/dim]" if model_line else "")
            + "\n[dim]Type 'exit' or Ctrl-C to quit.[/dim]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Health check
    if check_health(base_url):
        console.print(
            "[bold green]✓[/bold green] System server is reachable.\n"
        )
    else:
        console.print(
            f"[bold yellow]⚠[/bold yellow] System server not responding at "
            f"[bold]{base_url}[/bold].  Continuing anyway...\n"
        )

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        stream_prompt(
            base_url,
            user_input,
            router_model=args.router_model,
            generator_model=args.generator_model,
        )


if __name__ == "__main__":
    main()
