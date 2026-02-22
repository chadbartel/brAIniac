"""
brainiac/servers/system-server/src/system_server/orchestrator.py

Two-layer heterogeneous orchestrator.

Architecture:
  - Layer 1: ToolRouter (small, fast model — e.g. llama3.2:3b)
      Analyses the user prompt and conversation history.
      Outputs a single structured JSON routing decision:
          {"tools_needed": ["search_web"], "queries": ["..."]}
      Does NOT generate prose.  It is a pure routing controller.

  - Layer 2: Generator (larger model — e.g. dolphin-llama3)
      If tools were invoked, their results are injected into the final
      prompt inside <tool_results>...</tool_results> delimiters.
      Generates the user-facing answer.

Configure via environment variables:
  OLLAMA_BASE_URL          — OpenAI-compat Ollama endpoint
  OLLAMA_MODEL_ROUTER      — model tag for Layer 1 (default: llama3.2:3b)
  OLLAMA_MODEL_GENERATOR   — model tag for Layer 2 (default: dolphin-llama3)
  OLLAMA_MODEL             — legacy alias; if set, overrides OLLAMA_MODEL_GENERATOR
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx
from ddgs import DDGS
from pydantic import Field
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
    """Runtime configuration loaded from environment variables / .env file.

    Attributes:
        ollama_base_url: OpenAI-compatible Ollama base URL.
        ollama_model_router: Small model tag used for tool routing (Layer 1).
        ollama_model_generator: Larger model tag used for answer generation (Layer 2).
        ollama_model: Legacy alias.  If non-empty, overrides ollama_model_generator.
        research_server_url: SSE endpoint for the FastMCP research server.
        max_rounds: Maximum conversation rounds (kept very low to prevent loops).
        human_approval_keywords: Prompt keywords that trigger a human approval gate.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = Field(
        "http://ollama:11434/v1",
        description="OpenAI-compatible Ollama base URL.",
    )
    ollama_model_router: str = Field(
        "llama3.2:3b",
        description="Small, fast model used for tool routing (Layer 1).",
    )
    ollama_model_generator: str = Field(
        "dolphin-llama3",
        description="Larger model used for final answer generation (Layer 2).",
    )
    # Legacy alias — kept so existing .env files with OLLAMA_MODEL still work.
    ollama_model: str = Field(
        "",
        description=(
            "Deprecated.  Set OLLAMA_MODEL_ROUTER and OLLAMA_MODEL_GENERATOR "
            "instead.  If non-empty, overrides ollama_model_generator."
        ),
    )
    research_server_url: str = Field(
        "http://research-server:8100/sse",
        description="SSE endpoint for the FastMCP research server.",
    )
    max_rounds: int = Field(
        4,
        description=(
            "Maximum conversation rounds.  Kept low to prevent "
            "self-conversation hallucination loops."
        ),
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
            "If the prompt contains any of these keywords, the pipeline "
            "pauses for human approval before executing."
        ),
    )


cfg: OrchestratorSettings = OrchestratorSettings()

# ---------------------------------------------------------------------------
# Effective model names (legacy shim applied once at import time)
# ---------------------------------------------------------------------------

_ROUTER_MODEL: str = cfg.ollama_model_router
_GENERATOR_MODEL: str = cfg.ollama_model if cfg.ollama_model else cfg.ollama_model_generator

# ---------------------------------------------------------------------------
# Backward-compat LLM config dict (referenced by tests / REPL)
# ---------------------------------------------------------------------------

_LLM_CONFIG: dict[str, Any] = {
    "config_list": [
        {
            "model": _GENERATOR_MODEL,
            "base_url": cfg.ollama_base_url,
            "api_key": "ollama",
            "api_type": "openai",
        }
    ],
    "temperature": 0.4,
    "timeout": 120,
    "cache_seed": None,
}

# ---------------------------------------------------------------------------
# String-content normaliser
# ---------------------------------------------------------------------------


def _str_content(val: None | str | list[dict[str, Any]]) -> str:
    """Normalise an Ollama / AutoGen message content value to a plain string.

    Args:
        val: Raw content field — may be ``None``, a ``str``, or a list of
            dicts (multimodal / tool-call format).

    Returns:
        A plain string safe for all string operations.
    """
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts: list[str] = []
        for item in val:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(val)


# ---------------------------------------------------------------------------
# Answer cleaning helper
# ---------------------------------------------------------------------------

_ARTEFACT_PATTERN: re.Pattern[str] = re.compile(
    r"TERMINATE",
    re.IGNORECASE,
)


def _clean_answer(raw: str) -> str:
    """Strip residual orchestration artefacts from a final answer string.

    Removes any ``TERMINATE`` sentinels that legacy-prompted models may echo.

    Args:
        raw: Raw content string from a generator response.

    Returns:
        Human-readable answer with orchestration tokens removed.
    """
    return _ARTEFACT_PATTERN.sub("", raw).strip()


# ---------------------------------------------------------------------------
# MCP tool helpers — direct in-process execution
# ---------------------------------------------------------------------------


def _search_web_direct(query: str, max_results: int = 8) -> str:
    """Execute a DuckDuckGo search directly within the system-server process.

    Bypasses the research-server HTTP proxy entirely.  Small quantized models
    do not reliably emit OpenAI function-call JSON, so register_function()
    hooks never fire in practice.  Running the search here guarantees that
    real, current results are always available to the Generator.

    Args:
        query: The natural-language search query.
        max_results: Maximum number of results to return (default 8).

    Returns:
        JSON-encoded list of ``{title, url, snippet}`` dicts, or an error dict.
    """
    logger.info("[search_web_direct] query=%r max_results=%d", query, max_results)
    try:
        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": hit.get("title", ""),
                        "url": hit.get("href", ""),
                        "snippet": hit.get("body", ""),
                    }
                )
        logger.info("[search_web_direct] returned %d results", len(results))
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("[search_web_direct] DDG error: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})


def _call_research_tool(tool_name: str, **kwargs: Any) -> str:
    """Invoke a research tool by name, routing to the correct backend.

    ``search_web`` executes locally via DDGS.  Memory tools route through the
    research-server HTTP endpoint.

    Args:
        tool_name: One of ``"search_web"``, ``"store_memory"``, ``"query_memory"``.
        **kwargs: Tool-specific keyword arguments.

    Returns:
        JSON-encoded result string or error dict.
    """
    if tool_name == "search_web":
        return _search_web_direct(
            query=kwargs.get("query", ""),
            max_results=int(kwargs.get("max_results", 8)),
        )

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
        logger.error("[mcp-call] HTTP error for %s: %s", tool_name, exc, exc_info=True)
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.error(
            "[mcp-call] Unexpected error for %s: %s", tool_name, exc, exc_info=True
        )
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------


def _requires_human_approval(
    prompt: str, keywords: list[str]
) -> tuple[bool, str]:
    """Check whether the prompt contains any sensitive action keywords.

    Args:
        prompt: The full prompt string to scan.
        keywords: List of trigger keywords from settings.

    Returns:
        A tuple of (requires_approval, matched_keyword).
    """
    lower_prompt = prompt.lower()
    for kw in keywords:
        if kw.lower() in lower_prompt:
            return True, kw
    return False, ""


# ---------------------------------------------------------------------------
# Layer 1 — Tool Router
# ---------------------------------------------------------------------------

# Whitelist of tool names the router is permitted to request.
_KNOWN_TOOLS: frozenset[str] = frozenset({"search_web", "store_memory", "query_memory"})

_ROUTER_SYSTEM_PROMPT: str = (
    "You are a tool-routing controller. Your ONLY job is to decide which tools "
    "are needed to answer the user's message. Do NOT answer the question yourself.\n\n"
    "Output EXACTLY one JSON object and nothing else — no prose, no markdown, "
    "no code fences. The object MUST conform to this schema:\n\n"
    '{"tools_needed": ["search_web" | "store_memory" | "query_memory"], '
    '"queries": ["<one query string per tool>"]}\n\n'
    "Rules:\n"
    "- Set both lists to [] when the question can be answered from general "
    "knowledge without live data (greetings, maths, coding, grammar, creative "
    "writing, opinions, or explaining stable concepts).\n"
    '- Include \"search_web\" when the answer requires current events, live data, '
    "real-time prices, sports scores, recent news, or anything time-sensitive.\n"
    "- queries must have exactly one entry per tool in tools_needed.\n"
    '- If no tools are needed, output: {"tools_needed": [], "queries": []}'
)


@dataclasses.dataclass(slots=True)
class ToolRouterResult:
    """Structured output from the tool-routing layer (Layer 1).

    Attributes:
        tools_needed: Ordered list of MCP tool names to invoke.
        queries: One query string per entry in ``tools_needed``.
        raw: The raw JSON string returned by the router model.
    """

    tools_needed: list[str]
    queries: list[str]
    raw: str


def _call_tool_router(
    user_prompt: str,
    history: list[dict[str, Any]] | None = None,
    *,
    router_model: str | None = None,
) -> ToolRouterResult:
    """Call the router model (Layer 1) via the Ollama chat-completions API.

    Uses a direct ``httpx`` call rather than AutoGen to avoid GroupChat
    overhead and to enforce strict JSON-only output from the small model.
    History is passed as proper ``messages`` array entries so the model
    receives a structured multi-turn conversation rather than a text blob.
    On any parse failure the function returns a safe no-tool result so the
    pipeline never blocks.

    Args:
        user_prompt: The current user request (not including history).
        history: Prior conversation turns as
            ``[{"role": "user"|"assistant", "content": "..."}]``.
        router_model: Override the router model for this call.  Defaults to
            the ``OLLAMA_MODEL_ROUTER`` setting.

    Returns:
        A :class:`ToolRouterResult` with validated tool names and queries.
    """
    model: str = router_model or _ROUTER_MODEL
    completions_url: str = cfg.ollama_base_url.rstrip("/") + "/chat/completions"

    # Build messages: system prompt, then history turns, then current question.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _ROUTER_SYSTEM_PROMPT}
    ]
    for turn in (history or []):
        role = turn.get("role", "")
        content = _str_content(turn.get("content"))
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,  # deterministic routing decision
        "stream": False,
    }

    logger.info("[tool_router] model=%r url=%s", model, completions_url)
    try:
        response = httpx.post(completions_url, json=payload, timeout=60)
        response.raise_for_status()
        raw_text: str = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        logger.info("[tool_router] raw=%r", raw_text[:300])

        parsed: dict[str, Any] = json.loads(raw_text)
        tools_needed: list[str] = [
            t for t in parsed.get("tools_needed", []) if t in _KNOWN_TOOLS
        ]
        queries: list[str] = list(parsed.get("queries", []))

        # Normalise length: pad or truncate so len(queries) == len(tools_needed).
        if len(queries) < len(tools_needed):
            queries.extend([""] * (len(tools_needed) - len(queries)))
        else:
            queries = queries[: len(tools_needed)]

        result = ToolRouterResult(
            tools_needed=tools_needed, queries=queries, raw=raw_text
        )
        logger.info("[tool_router] tools_needed=%s", result.tools_needed)
        return result

    except json.JSONDecodeError as exc:
        logger.warning(
            "[tool_router] JSON parse failure (%s) — defaulting to no-tool path.", exc
        )
    except httpx.HTTPStatusError as exc:
        logger.error("[tool_router] HTTP error: %s", exc, exc_info=True)
    except Exception as exc:
        logger.error("[tool_router] Unexpected error: %s", exc, exc_info=True)

    # Safe fallback — proceed to direct generation without any tools.
    return ToolRouterResult(tools_needed=[], queries=[], raw="")


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tools(
    router_result: ToolRouterResult,
) -> list[dict[str, Any]]:
    """Execute the tools requested by the ToolRouter and collect their results.

    Args:
        router_result: The routing decision from :func:`_call_tool_router`.

    Returns:
        A list of dicts, one per tool invocation:
        ``{"tool": name, "query": query_string, "result": json_string}``.
    """
    tool_outputs: list[dict[str, Any]] = []
    for tool_name, query in zip(router_result.tools_needed, router_result.queries):
        logger.info(
            "[execute_tools] invoking tool=%r query=%r", tool_name, query
        )
        result_json: str = _call_research_tool(tool_name, query=query)
        tool_outputs.append(
            {"tool": tool_name, "query": query, "result": result_json}
        )
        logger.info(
            "[execute_tools] tool=%r returned %d chars",
            tool_name,
            len(result_json),
        )
    return tool_outputs


# ---------------------------------------------------------------------------
# Layer 2 — Generator
# ---------------------------------------------------------------------------

_GENERATOR_SYSTEM_PROMPT: str = (
    "You are brAIniac, a helpful and direct AI assistant. "
    "Answer the user's question accurately and concisely. "
    "Do NOT roleplay or simulate the user. "
    "Do NOT invent information not present in the provided context. "
    "Answer once and stop."
)

_GENERATOR_RESEARCH_SYSTEM_PROMPT: str = (
    "You are brAIniac, a helpful and direct AI assistant. "
    "You have been provided with real-time tool results inside "
    "<tool_results>...</tool_results> tags. "
    "Synthesise a clear, direct answer using ONLY those results. "
    "Do NOT fabricate data not present in the results. "
    "Do NOT say 'I must call a tool' or similar — the tools have already run. "
    "Do NOT simulate user responses. "
    "Answer once and stop."
)


def _call_generator(
    user_prompt: str,
    tool_outputs: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
    *,
    generator_model: str | None = None,
) -> str:
    """Call the generator model (Layer 2) via the Ollama chat-completions API.

    Conversation history is passed as proper ``messages`` array entries so
    the model natively understands multi-turn context and never echoes the
    input.  If tool results are available they are injected into the final
    user message inside ``<tool_results>...</tool_results>`` delimiters.

    Args:
        user_prompt: The current user request (not including history).
        tool_outputs: List of ``{tool, query, result}`` dicts from
            :func:`_execute_tools`.  May be empty for CHAT-mode requests.
        history: Prior conversation turns as
            ``[{"role": "user"|"assistant", "content": "..."}]``.
        generator_model: Override the generator model for this call.  Defaults
            to the ``OLLAMA_MODEL_GENERATOR`` setting.

    Returns:
        The raw response string from the generator model.

    Raises:
        httpx.HTTPStatusError: If the Ollama endpoint returns a non-2xx status.
    """
    model: str = generator_model or _GENERATOR_MODEL
    completions_url: str = cfg.ollama_base_url.rstrip("/") + "/chat/completions"

    system_content: str = (
        _GENERATOR_RESEARCH_SYSTEM_PROMPT if tool_outputs else _GENERATOR_SYSTEM_PROMPT
    )

    # Build messages: system, then history turns, then the final user message.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]
    for turn in (history or []):
        role = turn.get("role", "")
        content = _str_content(turn.get("content"))
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    if tool_outputs:
        # Inject tool results into the final user message only.
        sections: list[str] = []
        for entry in tool_outputs:
            sections.append(
                f"Tool: {entry['tool']}\n"
                f"Query: {entry['query']}\n"
                f"Results:\n{entry['result']}"
            )
        injection_block: str = "\n\n---\n\n".join(sections)
        final_user_content: str = (
            f"<tool_results>\n{injection_block}\n</tool_results>\n\n"
            f"User question: {user_prompt}"
        )
    else:
        final_user_content = user_prompt

    messages.append({"role": "user", "content": final_user_content})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "stream": False,
    }

    logger.info("[generator] model=%r url=%s", model, completions_url)
    response = httpx.post(completions_url, json=payload, timeout=120)
    response.raise_for_status()
    raw: str = (
        response.json()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    logger.info("[generator] response length=%d chars", len(raw))
    return raw


# ---------------------------------------------------------------------------
# Fallback helper
# ---------------------------------------------------------------------------


def _best_fallback(messages: list[dict[str, Any]]) -> str:
    """Return the longest non-empty message body from a message list.

    Used when the generator produces an empty response.

    Args:
        messages: List of message dicts with at least a ``content`` key.

    Returns:
        The most substantive message string, or a static sentinel.
    """
    candidates: list[str] = [
        _str_content(m.get("content"))
        for m in messages
        if m.get("role") != "system"
    ]
    return max(candidates, key=len, default="No response generated.")


# ---------------------------------------------------------------------------
# Internal event emitter
# ---------------------------------------------------------------------------


def _emit(
    on_message: Callable[[dict[str, Any]], None] | None,
    *,
    agent: str,
    recipient: str,
    content: str,
    event_type: str,
) -> None:
    """Fire the on_message callback if one is registered.

    Args:
        on_message: The optional callback registered by the API layer.
        agent: Name of the component emitting the event.
        recipient: Name of the receiving component.
        content: The message content string.
        event_type: Event kind — ``"message"`` or ``"function_call"``.
    """
    if on_message is None:
        return
    try:
        on_message(
            {
                "agent": agent,
                "recipient": recipient,
                "content": content,
                "type": event_type,
            }
        )
    except Exception as exc:
        logger.warning("on_message callback error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Public orchestration entry point
# ---------------------------------------------------------------------------


def run_task(
    user_prompt: str,
    on_message: Callable[[dict[str, Any]], None] | None = None,
    history: list[dict[str, Any]] | None = None,
    *,
    router_model: str | None = None,
    generator_model: str | None = None,
) -> str:
    """Execute a user task through the two-layer heterogeneous pipeline.

    Pipeline:
      1. Build full prompt with conversation history injected.
      2. Human approval gate — keyword check against the prompt.
      3. Layer 1 — ToolRouter: structured JSON routing decision.
      4. Tool execution — tools run in-process, results collected.
      5. Layer 2 — Generator: synthesises final answer, optionally grounded
         by injected ``<tool_results>`` context.

    Args:
        user_prompt: The raw user request.
        on_message: Optional callback fired for each pipeline event.  Receives
            a dict with keys ``agent``, ``recipient``, ``content``, ``type``.
        history: Prior conversation turns as
            ``[{"role": "user"|"assistant", "content": "..."}]``.
            Used to preserve multi-turn context across calls.
        router_model: Override the Layer 1 router model for this call.
            Defaults to ``OLLAMA_MODEL_ROUTER``.
        generator_model: Override the Layer 2 generator model for this call.
            Defaults to ``OLLAMA_MODEL_GENERATOR``.

    Returns:
        The final synthesised answer string.
    """
    logger.info("=== New task received ===")
    logger.info("Prompt: %s", user_prompt)
    if history:
        logger.info("History turns: %d", len(history))

    # -----------------------------------------------------------------------
    # Step 1 — Human approval gate (checks current prompt only)
    # -----------------------------------------------------------------------
    needs_approval, trigger = _requires_human_approval(
        user_prompt, cfg.human_approval_keywords
    )
    if needs_approval:
        logger.warning(
            "Prompt contains sensitive keyword %r — requesting human approval.",
            trigger,
        )
        print(
            f"\n[brAIniac] HUMAN APPROVAL REQUIRED\n"
            f"Trigger keyword: '{trigger}'\n"
            f"Prompt: {user_prompt}\n"
            f"Type 'yes' to proceed or anything else to abort: ",
            end="",
            flush=True,
        )
        user_response = input().strip().lower()
        if user_response != "yes":
            logger.info("Task aborted by human operator.")
            return "Task aborted by human operator."

    # -----------------------------------------------------------------------
    # Step 2 — Layer 1: ToolRouter
    # History is passed as structured messages — never serialised to a blob.
    # -----------------------------------------------------------------------
    _emit(
        on_message,
        agent="ToolRouter",
        recipient="Orchestrator",
        content=f"[Routing] Analysing prompt with {router_model or _ROUTER_MODEL}...",
        event_type="message",
    )
    router_result: ToolRouterResult = _call_tool_router(
        user_prompt, history, router_model=router_model
    )
    routing_summary: str = (
        f"[ToolRouter] tools_needed={router_result.tools_needed}  "
        f"queries={router_result.queries}"
    )
    logger.info(routing_summary)
    _emit(
        on_message,
        agent="ToolRouter",
        recipient="Orchestrator",
        content=routing_summary,
        event_type="message",
    )

    # -----------------------------------------------------------------------
    # Step 3 — Tool execution
    # -----------------------------------------------------------------------
    tool_outputs: list[dict[str, Any]] = []
    if router_result.tools_needed:
        tool_outputs = _execute_tools(router_result)
        for entry in tool_outputs:
            _emit(
                on_message,
                agent=entry["tool"],
                recipient="Orchestrator",
                content=(
                    f"[{entry['tool']}] query={entry['query']!r} "
                    f"({len(entry['result'])} chars returned)"
                ),
                event_type="function_call",
            )

    # -----------------------------------------------------------------------
    # Step 4 — Layer 2: Generator
    # History is passed as structured messages — never serialised to a blob.
    # -----------------------------------------------------------------------
    _emit(
        on_message,
        agent="Generator",
        recipient="User",
        content=f"[Generating] Synthesising with {generator_model or _GENERATOR_MODEL}...",
        event_type="message",
    )
    try:
        raw_answer: str = _call_generator(
            user_prompt,
            tool_outputs,
            history,
            generator_model=generator_model,
        )
    except httpx.HTTPStatusError as exc:
        logger.error("[generator] HTTP error: %s", exc, exc_info=True)
        raw_answer = ""
    except Exception as exc:
        logger.error("[generator] Unexpected error: %s", exc, exc_info=True)
        raw_answer = ""

    answer: str = _clean_answer(raw_answer)

    if not answer:
        logger.warning("[generator] Empty answer — using deterministic fallback.")
        answer = _best_fallback(
            [{"role": "user", "content": user_prompt}]
            + ([
                {"role": "assistant", "content": raw_answer}
            ] if raw_answer else [])
        )

    # Emit final answer through the standard agent channel so the stream
    # panel in the web tester shows the synthesised response.
    _emit(
        on_message,
        agent="OrchestratorAgent",
        recipient="chat_manager",
        content=answer,
        event_type="message",
    )

    logger.info("=== Task complete ===")
    return answer


# ---------------------------------------------------------------------------
# Interactive REPL entry point
# ---------------------------------------------------------------------------


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
