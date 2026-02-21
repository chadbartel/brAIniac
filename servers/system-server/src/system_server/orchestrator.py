"""
brainiac/servers/system-server/src/system_server/orchestrator.py

AutoGen (ag2) Orchestrator.

Architecture:
  - OrchestratorAgent (Manager)  — decomposes user tasks, builds a plan,
                                   and delegates to specialist workers.
  - ResearchAgent (Worker)       — connected to the FastMCP research-server
                                   (search_web, query_memory, store_memory).
  - HumanProxy                   — human-in-the-loop: approves sensitive
                                   actions before they execute.

The Ollama endpoint must expose an OpenAI-compatible /v1 API.
Set OLLAMA_BASE_URL and OLLAMA_MODEL via environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import queue
from collections.abc import Callable
from typing import Any

import httpx
from autogen import (
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    config_list_from_json,
)
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("system-server")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class OrchestratorSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = Field(
        "http://ollama:11434/v1",
        description="OpenAI-compatible Ollama base URL.",
    )
    ollama_model: str = Field(
        "dolphin-llama3",
        description="Ollama model tag to use for all agents.",
    )
    research_server_url: str = Field(
        "http://research-server:8100/sse",
        description="SSE endpoint for the FastMCP research server.",
    )
    max_rounds: int = Field(
        20,
        description="Maximum conversation rounds before auto-termination.",
    )
    human_approval_keywords: list[str] = Field(
        default_factory=lambda: [
            "delete",
            "remove",
            "deploy",
            "publish",
            "overwrite",
        ],
        description=(
            "If the agent plan contains these keywords, pause for "
            "human approval before proceeding."
        ),
    )


cfg: OrchestratorSettings = OrchestratorSettings()

# ---------------------------------------------------------------------------
# OpenAI-compatible LLM config (points at local Ollama)
# ---------------------------------------------------------------------------

_LLM_CONFIG: dict[str, Any] = {
    "config_list": [
        {
            "model": cfg.ollama_model,
            "base_url": cfg.ollama_base_url,
            "api_key": "ollama",  # placeholder; Ollama ignores this
            "api_type": "openai",
        }
    ],
    "temperature": 0.4,
    "timeout": 120,
    "cache_seed": None,  # disable caching for dynamic results
}

# ---------------------------------------------------------------------------
# MCP tool proxy helpers
# ---------------------------------------------------------------------------

# We call the research-server tools via raw HTTP+SSE in a function-calling
# shim, because ag2's native MCPToolkit requires an async event-loop that
# may conflict with GroupChatManager's sync execution path.  Switching to
# direct HTTP keeps the integration simple and auditable.


def _call_research_tool(tool_name: str, **kwargs: Any) -> str:
    """Invoke a tool on the research-server and return JSON-encoded result.

    Args:
        tool_name: One of "search_web", "store_memory", "query_memory".
        **kwargs: Tool-specific keyword arguments forwarded as JSON payload.

    Returns:
        JSON string of the tool response.
    """
    base = cfg.research_server_url.replace("/sse", "")
    url = f"{base}/tools/{tool_name}"
    logger.info("[mcp-call] tool=%s kwargs=%s", tool_name, kwargs)
    try:
        response = httpx.post(url, json=kwargs, timeout=30)
        response.raise_for_status()
        result = response.json()
        logger.info("[mcp-call] tool=%s -> %d bytes", tool_name, len(str(result)))
        return json.dumps(result, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[mcp-call] HTTP error for %s: %s", tool_name, exc, exc_info=True
        )
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.error(
            "[mcp-call] Unexpected error for %s: %s", tool_name, exc, exc_info=True
        )
        return json.dumps({"error": str(exc)})


# ag2 function-calling schemas exposed to the ResearchAgent
RESEARCH_FUNCTIONS: list[dict[str, Any]] = [
    {
        "name": "search_web",
        "description": (
            "Search the web using DuckDuckGo. Returns a list of "
            "{title, url, snippet} objects."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {
                    "type": "integer",
                    "default": 8,
                    "description": "Max results (1-20).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "store_memory",
        "description": "Store a text chunk with metadata in ChromaDB.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "metadata": {"type": "object", "default": {}},
                "doc_id": {
                    "type": "string",
                    "description": "Optional stable ID.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "query_memory",
        "description": "Semantic recall from ChromaDB.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]


def _dispatch_function(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a function call from an agent to the correct MCP tool."""
    if name in {"search_web", "store_memory", "query_memory"}:
        return _call_research_tool(name, **arguments)
    return json.dumps({"error": f"Unknown function: {name}"})


# ---------------------------------------------------------------------------
# Streaming hook
# ---------------------------------------------------------------------------


def _attach_stream_hook(
    agents: list[Any],
    on_message: Callable[[dict[str, Any]], None],
) -> None:
    """Monkey-patch ``_print_received_message`` on every agent so each
    incoming message also fires the *on_message* callback.

    Args:
        agents: AutoGen agent instances to instrument.
        on_message: Callback receiving a dict with keys ``agent`` (sender
            name), ``recipient`` (receiver name), ``content`` (text), and
            ``type`` (``"message"`` or ``"function_call"``).
    """
    for agent in agents:
        original = agent._print_received_message

        def _hook(
            message: dict[str, Any] | str,
            sender: Any,
            *,
            _original: Any = original,
            _recipient: Any = agent,
        ) -> None:
            _original(message, sender)
            if isinstance(message, str):
                content: str = message
                event_type: str = "message"
            else:
                fc = message.get("function_call")
                if fc:
                    content = (
                        f"[tool: {fc.get('name')}] "
                        f"{fc.get('arguments', '')}"
                    )
                    event_type = "function_call"
                else:
                    content = message.get("content") or ""
                    event_type = "message"
            on_message(
                {
                    "agent": sender.name,
                    "recipient": _recipient.name,
                    "content": content,
                    "type": event_type,
                }
            )

        agent._print_received_message = _hook  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _build_orchestrator_agent() -> AssistantAgent:
    """Create the Manager / Orchestrator agent.

    Responsibilities:
      - Analyse the user's high-level prompt.
      - Produce a numbered execution plan.
      - Delegate each step by addressing the appropriate worker agent.
      - Emit ``TERMINATE`` when the task is fully resolved.
      - Emit ``HUMAN_APPROVAL_REQUIRED: <reason>`` when a destructive or
        irreversible action is about to be taken.
    """
    system_message = """You are the brAIniac Orchestrator.

Your responsibilities:
1. Analyse the user's request thoroughly.
2. Decompose it into a clear, numbered execution plan.
3. Delegate each step to the most suitable agent:
   - ResearchAgent : web search, memory storage/retrieval.
4. Synthesise all results into a coherent final response.
5. Before taking any action described as: delete, remove, deploy, publish,
   or overwrite — emit exactly: HUMAN_APPROVAL_REQUIRED: <brief reason>.
6. When the full task is complete, emit exactly: TERMINATE

Rules:
- Never invent facts. Use ResearchAgent to fetch real-time data.
- Keep your plan concise; avoid unnecessary steps.
- Log your reasoning before each delegation.
"""
    return AssistantAgent(
        name="OrchestratorAgent",
        system_message=system_message,
        llm_config=_LLM_CONFIG,
        human_input_mode="NEVER",
    )


def _build_research_agent() -> ConversableAgent:
    """Create the ResearchAgent with access to all research-server tools."""
    system_message = """You are the brAIniac ResearchAgent.

You have access to three tools:
  - search_web(query, max_results): live DuckDuckGo search.
  - store_memory(text, metadata, doc_id): persist findings to ChromaDB.
  - query_memory(query, top_k): recall similar past findings.

When asked to research a topic:
1. First query_memory to avoid redundant searches.
2. If memory is insufficient, call search_web.
3. Always store_memory for important new findings.
4. Return a concise, cited summary to the OrchestratorAgent.
"""
    agent = ConversableAgent(
        name="ResearchAgent",
        system_message=system_message,
        llm_config={
            **_LLM_CONFIG,
            "functions": RESEARCH_FUNCTIONS,
        },
        human_input_mode="NEVER",
    )

    # Register function implementations
    for fn in RESEARCH_FUNCTIONS:
        fn_name: str = fn["name"]
        agent.register_function(
            function_map={
                fn_name: lambda arguments, _name=fn_name: _dispatch_function(
                    _name, arguments
                )
            }
        )

    return agent


def _build_human_proxy() -> UserProxyAgent:
    """Create the HumanProxy agent for human-in-the-loop checkpoints."""
    return UserProxyAgent(
        name="HumanProxy",
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config=False,
        system_message=(
            "You represent the human operator. "
            "Approve or reject actions flagged with HUMAN_APPROVAL_REQUIRED."
        ),
    )


# ---------------------------------------------------------------------------
# Termination condition
# ---------------------------------------------------------------------------


def _is_terminal(message: dict[str, Any]) -> bool:
    """Return True when the conversation should end."""
    content: str = message.get("content", "") or ""
    return "TERMINATE" in content


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------


def _requires_human_approval(
    plan: str, keywords: list[str]
) -> tuple[bool, str]:
    """Check whether the plan contains any sensitive action keywords.

    Args:
        plan: The stringified orchestrator plan.
        keywords: List of trigger keywords from settings.

    Returns:
        A tuple of (requires_approval, matched_keyword).
    """
    lower_plan = plan.lower()
    for kw in keywords:
        if kw.lower() in lower_plan:
            return True, kw
    return False, ""


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def run_task(
    user_prompt: str,
    on_message: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Execute a user task through the multi-agent orchestration pipeline.

    Args:
        user_prompt: The raw user request.
        on_message: Optional callback invoked for every agent message or
            tool call.  Receives a dict with keys ``agent``, ``recipient``,
            ``content``, and ``type``.

    Returns:
        The final synthesised answer from the OrchestratorAgent.

    Raises:
        RuntimeError: If the orchestration pipeline fails.
    """
    logger.info("=== New task received ===")
    logger.info("Prompt: %s", user_prompt)

    orchestrator = _build_orchestrator_agent()
    researcher = _build_research_agent()
    human = _build_human_proxy()

    if on_message is not None:
        _attach_stream_hook([orchestrator, researcher, human], on_message)

    # Pre-flight approval check: ask the orchestrator to outline its plan
    # first, then gate on sensitive keywords before entering the group chat.
    planning_config: dict[str, Any] = {**_LLM_CONFIG}
    planning_agent = AssistantAgent(
        name="_PlanningProbe",
        system_message=(
            "Briefly list (one sentence each) the high-level steps you would "
            "take to fulfil the following request. Do not execute anything."
        ),
        llm_config=planning_config,
        human_input_mode="NEVER",
    )
    planning_proxy = UserProxyAgent(
        name="_PlanningProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )
    planning_proxy.initiate_chat(
        planning_agent, message=user_prompt, max_turns=1, silent=True
    )
    raw_plan = (
        planning_proxy.last_message(planning_agent) or {}
    ).get("content", "")

    needs_approval, trigger = _requires_human_approval(
        raw_plan, cfg.human_approval_keywords
    )
    if needs_approval:
        logger.warning(
            "Plan contains sensitive keyword %r — requesting human approval",
            trigger,
        )
        print(
            f"\n[brAIniac] HUMAN APPROVAL REQUIRED\n"
            f"Trigger keyword: '{trigger}'\n"
            f"Proposed plan:\n{raw_plan}\n"
            f"Type 'yes' to proceed or anything else to abort: ",
            end="",
            flush=True,
        )
        user_response = input().strip().lower()
        if user_response != "yes":
            logger.info("Task aborted by human operator.")
            return "Task aborted by human operator."

    # --- Main group chat ---
    group_chat = GroupChat(
        agents=[orchestrator, researcher, human],
        messages=[],
        max_round=cfg.max_rounds,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=_LLM_CONFIG,
        is_termination_msg=_is_terminal,
    )

    logger.info("Starting group chat with %d max rounds", cfg.max_rounds)
    human.initiate_chat(manager, message=user_prompt, silent=False)

    # Extract the final answer from the last orchestrator message
    final_messages = [
        m
        for m in group_chat.messages
        if m.get("name") == "OrchestratorAgent"
    ]
    final_answer = (
        final_messages[-1].get("content", "No result produced.")
        if final_messages
        else "No result produced."
    )
    # Strip the TERMINATE marker if present
    final_answer = final_answer.replace("TERMINATE", "").strip()

    logger.info("=== Task complete ===")
    return final_answer


def run() -> None:
    """Interactive REPL entry point (used by the Poetry script)."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    console.print(
        Panel(
            "[bold cyan]brAIniac System Server[/bold cyan]\n"
            "[dim]Local-first AI Orchestrator — type 'exit' to quit[/dim]",
            border_style="cyan",
        )
    )

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        if not user_input:
            continue
        try:
            result = run_task(user_input)
            console.print(Panel(result, title="brAIniac", border_style="green"))
        except Exception as exc:
            logger.error("Orchestration error: %s", exc, exc_info=True)
            console.print(f"[bold red]Error:[/bold red] {exc}")


if __name__ == "__main__":
    run()
