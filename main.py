#!/usr/bin/env python3
"""main.py

Entry point for brAIniac - Local-first, uncensored AI chatbot.
Provides an interactive CLI interface using the Rich library.
"""

from __future__ import annotations

# Standard Library
import os
import sys
from typing import NoReturn

# Third-Party Libraries
from dotenv import load_dotenv
from rich.panel import Panel
from rich.theme import Theme
from rich.prompt import Prompt
from rich.console import Console
from rich.markdown import Markdown

# Local Modules
from core.chat import ChatEngine

# Load environment variables from .env file
load_dotenv()

# Initialize Rich console with custom theme
custom_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "user": "bold blue",
        "assistant": "green",
    }
)
console = Console(theme=custom_theme)


def display_banner() -> None:
    """Display the brAIniac welcome banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗ ██████╗  █████╗ ██╗███╗   ██╗██╗ █████╗  ██████╗║
    ║   ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██║██╔══██╗██╔════╝║
    ║   ██████╔╝██████╔╝███████║██║██╔██╗ ██║██║███████║██║     ║
    ║   ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║██║██╔══██║██║     ║
    ║   ██████╔╝██║  ██║██║  ██║██║██║ ╚████║██║██║  ██║╚██████╗║
    ║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝ ╚═════╝║
    ║                                                           ║
    ║        Local-First, Uncensored AI Chatbot - Phase 1       ║
    ║              Optimized for 8GB VRAM (RTX 2070S)           ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")
    console.print()


def display_help() -> None:
    """Display available commands and usage information."""
    help_text = """
**Available Commands:**

- `/help` - Show this help message
- `/clear` - Clear conversation history
- `/stats` - Show current context statistics
- `/quit` or `/exit` - Exit brAIniac
- Any other text - Chat with the AI

**Tips:**

- Keep conversations focused to maintain context quality
- Use `/clear` if responses become inconsistent
- The rolling memory window keeps the last 20 messages
    """
    console.print(Panel(Markdown(help_text), title="Help", border_style="cyan"))


def display_stats(engine: ChatEngine) -> None:
    """Display current context and memory statistics.

    Args:
        engine: The ChatEngine instance.
    """
    msg_count = engine.get_message_count()
    max_msgs = engine.memory.max_messages

    stats_text = f"""
**Context Statistics:**

- Messages in context: {msg_count}/{max_msgs}
- Model: `{engine.model}`
- Ollama host: `{engine.ollama_host}`
- Context utilization: {(msg_count / max_msgs * 100):.1f}%
    """
    console.print(Panel(Markdown(stats_text), title="Statistics", border_style="cyan"))


def main() -> NoReturn:
    """Main entry point for the brAIniac CLI."""
    # Display welcome banner
    display_banner()

    # Get configuration from environment variables
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    max_messages = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))

    # Show startup information
    console.print("⚙️  Initializing brAIniac...", style="info")
    console.print(f"📍 Ollama host: {ollama_host}", style="info")
    console.print(f"🤖 Model: {model}", style="info")
    console.print(f"💾 Rolling memory: {max_messages} messages\n", style="info")

    # Initialize chat engine
    try:
        engine = ChatEngine(
            model=model,
            ollama_host=ollama_host,
            max_context_messages=max_messages,
        )
        console.print("✅ brAIniac ready!\n", style="success")
    except Exception as exc:
        console.print(f"❌ Failed to initialize: {exc}", style="error")
        console.print(
            "\nMake sure Ollama is running and the model is downloaded.",
            style="warning",
        )
        console.print(f"Try running: ollama pull {model}\n", style="info")
        sys.exit(1)

    # Display quick help
    console.print(
        "Type [bold]/help[/bold] for commands, or start chatting!\n", style="info"
    )

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                console.print("\n👋 Goodbye!\n", style="success")
                sys.exit(0)

            elif user_input.lower() == "/help":
                display_help()
                continue

            elif user_input.lower() == "/clear":
                engine.clear_history()
                console.print("🗑️  Conversation history cleared.\n", style="success")
                continue

            elif user_input.lower() == "/stats":
                display_stats(engine)
                continue

            # Process chat message
            console.print()  # Blank line for readability
            with console.status("[bold green]Thinking...", spinner="dots"):
                response = engine.chat(user_input)

            # Display assistant response
            console.print(
                Panel(
                    Markdown(response),
                    title="[bold green]brAIniac[/bold green]",
                    border_style="green",
                )
            )
            console.print()  # Blank line after response

        except KeyboardInterrupt:
            console.print("\n\n👋 Interrupted. Goodbye!\n", style="warning")
            sys.exit(0)

        except Exception as exc:
            console.print(f"\n❌ Error: {exc}\n", style="error")
            console.print(
                "You can continue chatting or type /quit to exit.\n", style="info"
            )


if __name__ == "__main__":
    main()
