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
import re
from collections.abc import Callable
from typing import Any

import httpx
from ddgs import DDGS
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

from .intent_classifier import Mode, extract_clarify_question, extract_mode

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
        6,
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


def _search_web_direct(query: str, max_results: int = 8) -> str:
    """Execute a DuckDuckGo search directly within the system-server process.

    This bypasses the research-server HTTP proxy entirely.  The quantized
    dolphin-llama3 model does not reliably emit OpenAI function-call JSON, so
    ``register_function()`` hooks on the ResearchAgent never fire in practice.
    By running the search here — before the GroupChat starts — we guarantee
    that real, current results are always injected into the agent context.

    Args:
        query: The user's natural-language search query.
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
        logger.error(
            "[search_web_direct] DuckDuckGo error: %s", exc, exc_info=True
        )
        return json.dumps({"error": str(exc)})


def _call_research_tool(tool_name: str, **kwargs: Any) -> str:
    """Invoke a tool on the research-server and return JSON-encoded result.

    ``search_web`` is executed locally via DDGS to avoid depending on the
    research-server's MCP-over-SSE transport, which has no plain HTTP API.
    Memory tools (``store_memory``, ``query_memory``) still route through the
    research-server HTTP endpoint if one is available.

    Args:
        tool_name: One of "search_web", "store_memory", "query_memory".
        **kwargs: Tool-specific keyword arguments forwarded as JSON payload.

    Returns:
        JSON string of the tool response.
    """
    if tool_name == "search_web":
        return _search_web_direct(
            query=kwargs.get("query", ""),
            max_results=int(kwargs.get("max_results", 8)),
        )

    # Memory tools: attempt the research-server HTTP endpoint.
    base = cfg.research_server_url.replace("/sse", "")
    url = f"{base}/tools/{tool_name}"
    logger.info("[mcp-call] tool=%s kwargs=%s", tool_name, kwargs)
    try:
        response = httpx.post(url, json=kwargs, timeout=30)
        response.raise_for_status()
        result = response.json()
        logger.info(
            "[mcp-call] tool=%s -> %d bytes", tool_name, len(str(result))
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[mcp-call] HTTP error for %s: %s", tool_name, exc, exc_info=True
        )
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        logger.error(
            "[mcp-call] Unexpected error for %s: %s",
            tool_name,
            exc,
            exc_info=True,
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
            try:
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
                if content:  # skip empty keep-alive messages
                    on_message(
                        {
                            "agent": sender.name,
                            "recipient": _recipient.name,
                            "content": content,
                            "type": event_type,
                        }
                    )
            except Exception as _hook_exc:
                logger.warning(
                    "Stream hook error: %s", _hook_exc, exc_info=True
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
    system_message = (
        "You are brAIniac, a versatile AI assistant. For every message,\n"
        "first decide the MODE, then act accordingly.\n\n"
        "────────────────────────────────────────\n"
        "MODE: CHAT\n"
        "  Triggers: greetings, casual conversation, opinions, general\n"
        "  knowledge you are confident is stable (e.g. maths, grammar,\n"
        "  coding questions, explaining concepts, writing tasks).\n"
        "  Action: Answer directly and helpfully. End with TERMINATE.\n\n"
        "MODE: RESEARCH\n"
        "  Triggers: current events, live data, statistics, news, prices,\n"
        "  sports scores, anything where your training data may be stale\n"
        "  or the user explicitly asks for up-to-date information.\n"
        "  Action:\n"
        "    1. Send exactly: ResearchAgent, please search for: <query>\n"
        "    2. Wait for ResearchAgent to return real results.\n"
        "    3. Synthesise a clear answer using ONLY the returned data.\n"
        "    4. End with TERMINATE.\n\n"
        "MODE: CLARIFY\n"
        "  Triggers: the query is genuinely ambiguous — a key term could\n"
        "  refer to multiple distinct things and the wrong interpretation\n"
        "  would produce a useless answer.\n"
        "  Action: Ask ONE specific question naming the concrete\n"
        "  alternatives (e.g. \"'Jaguar' — the car brand, the animal, or\n"
        "  the macOS version?\"). NEVER ask vague questions like 'What is\n"
        "  the focus of your interest?'.\n"
        "  Format:\n"
        "    CLARIFY: <question naming the alternatives>\n"
        "    TERMINATE\n\n"
        "────────────────────────────────────────\n"
        "RULES (apply to all modes):\n"
        "- Choose exactly ONE mode per message. Do NOT combine them.\n"
        "- Do NOT roleplay or simulate what other agents would say.\n"
        "- Do NOT fabricate data. If ResearchAgent returns nothing, say so.\n"
        "- For destructive actions (delete/remove/deploy/publish/overwrite)\n"
        "  emit: HUMAN_APPROVAL_REQUIRED: <reason>\n"
        "- Always end your final response with TERMINATE on its own line.\n\n"
        "────────────────────────────────────────\n"
        "STRICT PROHIBITIONS:\n"
        "- NEVER write '[User responds]' or pretend to speak as the user.\n"
        "- NEVER simulate a multi-turn dialogue or invent follow-up exchanges.\n"
        "- NEVER write what the user might say or ask next.\n"
        "- Answer the CURRENT question EXACTLY ONCE, then TERMINATE immediately.\n"
        "- If this message contains [Real-time search results], synthesise from\n"
        "  them DIRECTLY — do NOT re-delegate to ResearchAgent.\n"
    )
    return AssistantAgent(
        name="OrchestratorAgent",
        system_message=system_message,
        llm_config=_LLM_CONFIG,
        human_input_mode="NEVER",
    )


def _build_research_agent() -> ConversableAgent:
    """Create the ResearchAgent with access to all research-server tools."""
    system_message = (
        "You are the brAIniac ResearchAgent. You have tools — USE THEM.\n\n"
        "TOOLS:\n"
        "  search_web(query, max_results=8): Live DuckDuckGo web search. "
        "Call this for ANY factual or current information.\n"
        "  store_memory(text): Persist findings to ChromaDB.\n"
        "  query_memory(query, top_k=5): Recall past findings.\n\n"
        "PROCESS:\n"
        "1. When asked to search, call search_web IMMEDIATELY with the query.\n"
        "2. Return the verbatim titles, URLs, and snippets from the results.\n"
        "3. If search_web returns no results, report that — never fabricate data.\n\n"
        "CRITICAL:\n"
        "- You MUST call search_web. Never answer from your own knowledge.\n"
        "- After returning results, DO NOT ask follow-up questions like \'Is\n"
        "  there anything else you would like to know?\'. Your job is ONLY to\n"
        "  fetch and return data. OrchestratorAgent will handle the response.\n"
    )
    agent = ConversableAgent(
        name="ResearchAgent",
        system_message=system_message,
        llm_config={
            **_LLM_CONFIG,
            "functions": RESEARCH_FUNCTIONS,
        },
        human_input_mode="NEVER",
    )

    # Register function implementations.
    # Use a factory to capture fn_name correctly and accept **kwargs as ag2
    # calls registered functions via func(**parsed_json_arguments).
    def _make_tool(name: str) -> Callable[..., str]:
        def _fn(**kwargs: Any) -> str:
            return _dispatch_function(name, kwargs)

        _fn.__name__ = name
        return _fn

    for fn in RESEARCH_FUNCTIONS:
        fn_name: str = fn["name"]
        agent.register_function(function_map={fn_name: _make_tool(fn_name)})

    return agent


def _build_human_proxy() -> UserProxyAgent:
    """Create the HumanProxy agent for human-in-the-loop checkpoints."""
    return UserProxyAgent(
        name="HumanProxy",
        human_input_mode="NEVER",  # headless API — no TTY/stdin available
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
# Mode probe
# ---------------------------------------------------------------------------


def _probe_mode(prompt: str) -> tuple[Mode, str]:
    """Run a single-turn call using the OrchestratorAgent's system message to
    determine the intended mode for *prompt* before spinning up GroupChat.

    This lets CLARIFY responses short-circuit the full multi-agent pipeline
    (saving up to 20 GroupChat rounds) while keeping all routing logic inside
    the LLM — no domain rules live here.

    Args:
        prompt: The full user prompt (may include conversation history prefix).

    Returns:
        A tuple of (``Mode``, raw LLM output string).
    """
    probe_agent = _build_orchestrator_agent()
    probe_proxy = UserProxyAgent(
        name="_ModeProbeProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=1,
        code_execution_config=False,
    )
    probe_proxy.initiate_chat(
        probe_agent,
        message=(
            f"{prompt}\n\n"
            "[System note: Respond with your chosen MODE token only "
            "(CHAT / RESEARCH / CLARIFY) followed by any required content. "
            "Do NOT execute any actions.]"
        ),
        max_turns=1,
        silent=True,
    )
    raw: str = (probe_proxy.last_message(probe_agent) or {}).get("content", "")
    return extract_mode(raw), raw


# ---------------------------------------------------------------------------
# Answer cleaning helper
# ---------------------------------------------------------------------------

_MODE_PREFIX: re.Pattern[str] = re.compile(
    r"^\s*MODE:\s*(?:CHAT|RESEARCH|CLARIFY)\s*\n?",
    re.IGNORECASE,
)


def _str_content(val: Any) -> str:
    """Normalise an AutoGen message ``content`` value to a plain string.

    AutoGen ag2 stores ``content`` as ``None``, a ``str``, or a ``list`` of
    dicts (multimodal / tool-call format).  This helper collapses all forms
    so that string operations like ``.strip()`` never raise a TypeError.

    Args:
        val: The raw value of a message ``content`` field.

    Returns:
        A plain string representation safe for all string operations.
    """
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        # Flatten list-of-dicts (tool-call / multimodal): extract text fields.
        parts: list[str] = []
        for item in val:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return " ".join(p for p in parts if p)
    return str(val)


def _clean_answer(raw: str) -> str:
    """Strip orchestration artefacts from a final answer string.

    Removes the ``MODE: <TOKEN>`` prefix that the OrchestratorAgent emits
    as part of its mode declaration, and the ``TERMINATE`` sentinel.

    Args:
        raw: Raw content string from an OrchestratorAgent message.

    Returns:
        Human-readable answer with all orchestration tokens removed.
    """
    text = raw.replace("TERMINATE", "").strip()
    text = _MODE_PREFIX.sub("", text).strip()
    return text


def run_task(
    user_prompt: str,
    on_message: Callable[[dict[str, Any]], None] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Execute a user task through the multi-agent orchestration pipeline.

    Args:
        user_prompt: The raw user request.
        on_message: Optional callback invoked for every agent message or
            tool call.  Receives a dict with keys ``agent``, ``recipient``,
            ``content``, and ``type``.
        history: Optional list of prior conversation turns in
            ``{"role": "user"|"assistant", "content": "..."}`` format.
            When provided, the full conversation context is prepended to
            ``user_prompt`` so agents have memory across turns.

    Returns:
        The final synthesised answer from the OrchestratorAgent.

    Raises:
        RuntimeError: If the orchestration pipeline fails.
    """
    logger.info("=== New task received ===")
    logger.info("Prompt: %s", user_prompt)

    # Build full_prompt: prepend conversation history so the orchestrator
    # understands follow-up messages (e.g. disambiguation replies).
    if history:
        context_lines: list[str] = ["[Conversation so far]"]
        for turn in history:
            role_label = "User" if turn.get("role") == "user" else "Assistant"
            context_lines.append(f"{role_label}: {_str_content(turn.get('content')).strip()}")
        context_lines.append("[Current question]")
        full_prompt: str = "\n".join(context_lines) + "\n" + user_prompt
        logger.info("Injected %d prior history turns into prompt", len(history))
    else:
        full_prompt = user_prompt

    orchestrator = _build_orchestrator_agent()
    researcher = _build_research_agent()
    human = _build_human_proxy()

    # Stream hook is attached AFTER manager is created so the manager is
    # also patched — GroupChatManager is the actual recipient of every
    # agent → chat_manager message and must be included.

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
        planning_agent, message=full_prompt, max_turns=1, silent=True
    )
    raw_plan = (planning_proxy.last_message(planning_agent) or {}).get(
        "content", ""
    )

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

    # --- Intent classification: probe mode before spinning up GroupChat ---
    # A single-turn call with the OrchestratorAgent's system message detects
    # CLARIFY early, avoiding wasted GroupChat rounds. CHAT and RESEARCH both
    # proceed to GroupChat where the agent has full streaming context.
    detected_mode, mode_raw = _probe_mode(full_prompt)
    logger.info("Detected mode: %s", detected_mode)

    if detected_mode is Mode.CLARIFY:
        question = extract_clarify_question(mode_raw)
        if question:
            logger.info(
                "=== Task complete (early CLARIFY — GroupChat skipped) ==="
            )
            return f"I need a bit more information: {question}"
        # Malformed CLARIFY output — fall through to GroupChat as a safe default.
        logger.warning(
            "CLARIFY mode detected but question extraction failed; "
            "falling through to GroupChat."
        )
    elif detected_mode is Mode.UNKNOWN:
        logger.warning(
            "Mode probe returned UNKNOWN — proceeding to GroupChat as fallback."
        )
    # CHAT and RESEARCH both proceed; OrchestratorAgent handles them inside
    # GroupChat where it can stream its reasoning and call tools as needed.

    # --- Pre-execute search for RESEARCH / UNKNOWN mode ---
    # dolphin-llama3 (quantized 8B) does not reliably emit OpenAI
    # function-call JSON, so register_function() never fires.  We execute
    # search_web ourselves, then perform a direct single-turn synthesis call
    # (same pattern as the CLARIFY short-circuit) and return the answer
    # without entering GroupChat at all.  Injecting several KB of JSON into
    # GroupChat's initial message overloads the speaker-selection LLM and
    # causes it to never select OrchestratorAgent.
    chat_prompt: str = full_prompt
    if detected_mode in (Mode.RESEARCH, Mode.UNKNOWN):
        logger.info(
            "Pre-executing search_web for RESEARCH mode (mode=%s)", detected_mode
        )
        results_json: str = _call_research_tool(
            "search_web", query=user_prompt, max_results=8
        )
        synthesis_prompt: str = (
            f"User question: {user_prompt}\n\n"
            "Real-time search results (already fetched for you):\n"
            f"{results_json}\n\n"
            "Synthesise a direct, accurate answer using ONLY the search results "
            "above.  Do NOT say 'I must call search_web' or similar — the search "
            "is already done.  Do NOT invent information not present in the "
            "results.  Do NOT simulate user responses.  Answer once, then "
            "TERMINATE."
        )
        synth_agent = _build_orchestrator_agent()
        synth_proxy = UserProxyAgent(
            name="_ResearchSynthProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            code_execution_config=False,
        )
        synth_proxy.initiate_chat(
            synth_agent,
            message=synthesis_prompt,
            max_turns=1,
            silent=True,
        )
        raw_synth: str = _str_content(
            (synth_proxy.last_message(synth_agent) or {}).get("content")
        )
        research_answer: str = _clean_answer(raw_synth)
        if research_answer:
            logger.info("=== Task complete (RESEARCH direct synthesis) ===")
            # Fire the stream callback so the Thought Process panel shows
            # the agent's reasoning even though GroupChat was skipped.
            if on_message is not None:
                on_message(
                    {
                        "agent": "OrchestratorAgent",
                        "recipient": "chat_manager",
                        "content": raw_synth,
                        "type": "message",
                    }
                )
            return research_answer
        logger.warning(
            "RESEARCH direct synthesis produced empty answer — "
            "falling through to GroupChat."
        )

    # --- Main group chat ---
    group_chat = GroupChat(
        agents=[orchestrator, researcher, human],
        messages=[],
        max_round=cfg.max_rounds,
        speaker_selection_method="auto",
        allow_repeat_speaker=True,  # orchestrator must speak again after researcher returns results
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=_LLM_CONFIG,
        is_termination_msg=_is_terminal,
    )

    # Attach hook here so GroupChatManager is also patched — it is the
    # recipient of every outbound agent message in a GroupChat.
    if on_message is not None:
        _attach_stream_hook(
            [orchestrator, researcher, human, manager], on_message
        )

    logger.info("Starting group chat with %d max rounds", cfg.max_rounds)
    human.initiate_chat(manager, message=chat_prompt, silent=False)

    # ---------------------------------------------------------------------------
    # Extract the final answer from the group chat transcript.
    # Priority order:
    #   1. OrchestratorAgent CLARIFY message  → return as a question to the user
    #   2. OrchestratorAgent synthesis message → the real answer
    #   3. Last ResearchAgent message           → fallback if orchestrator timed out
    #   4. Last message from any agent          → last resort
    # A "delegation" message (starts with "ResearchAgent,") is intentionally
    # excluded from synthesis candidates — it is the task assignment, not the answer.
    # ---------------------------------------------------------------------------
    all_messages: list[dict[str, Any]] = group_chat.messages

    orch_messages: list[dict[str, Any]] = [
        m for m in all_messages if m.get("name") == "OrchestratorAgent"
    ]

    # Detect a clarification request emitted by the orchestrator.
    clarify_messages: list[dict[str, Any]] = [
        m for m in orch_messages if "CLARIFY:" in _str_content(m.get("content"))
    ]
    if clarify_messages:
        raw_clarify: str = _str_content(clarify_messages[-1].get("content"))
        # Extract just the question text after the "CLARIFY:" prefix.
        clarify_text: str = (
            raw_clarify.split("CLARIFY:", 1)[-1]
            .replace("TERMINATE", "")
            .strip()
        )
        logger.info("Orchestrator requested clarification: %s", clarify_text)
        logger.info("=== Task complete (clarification requested) ===")
        return f"I need a bit more information to help you accurately: {clarify_text}"

    # Synthesis messages: OrchestratorAgent messages that are NOT pure delegations.
    # A delegation message begins with the ResearchAgent's name followed by a comma.
    synthesis_messages: list[dict[str, Any]] = [
        m
        for m in orch_messages
        if not _str_content(m.get("content")).strip().startswith("ResearchAgent,")
    ]

    if synthesis_messages:
        # Walk in reverse: the model sometimes emits TERMINATE as a stand-alone
        # second message, which strips to "". Skip those and take the last
        # message that has real content after cleaning.
        for msg in reversed(synthesis_messages):
            final_answer = _clean_answer(_str_content(msg.get("content")))
            if final_answer:
                logger.info("=== Task complete ===")
                return final_answer

    # OrchestratorAgent never synthesized — fall back to last ResearchAgent message.
    # Pick the most substantive message (longest content) rather than the last,
    # since the last message is often a short conversational closer.
    researcher_messages: list[dict[str, Any]] = [
        m for m in all_messages if m.get("name") == "ResearchAgent"
        and len(_str_content(m.get("content")).strip()) > 80  # exclude short closers
    ]
    if not researcher_messages:
        # Widen the net if the threshold eliminated everything
        researcher_messages = [
            m for m in all_messages if m.get("name") == "ResearchAgent"
        ]
    if researcher_messages:
        best = max(researcher_messages, key=lambda m: len(_str_content(m.get("content"))))
        fallback = _str_content(best.get("content")).replace("TERMINATE", "").strip()
        if fallback:
            logger.warning(
                "OrchestratorAgent produced no synthesis — returning last "
                "ResearchAgent message as fallback."
            )
            logger.info("=== Task complete (researcher fallback) ===")
            return fallback

    # Absolute last resort: last message from any participant.
    if all_messages:
        last_content = (
            _str_content(all_messages[-1].get("content"))
        ).replace("TERMINATE", "").strip()
        if last_content:
            logger.warning("Using last chat message as final answer (last resort).")
            logger.info("=== Task complete (last-resort fallback) ===")
            return last_content

    logger.warning("No usable answer produced by any agent.")
    logger.info("=== Task complete (empty) ===")
    return "I wasn't able to find a result for that query. Could you provide more detail?"


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
            console.print(
                Panel(result, title="brAIniac", border_style="green")
            )
        except Exception as exc:
            logger.error("Orchestration error: %s", exc, exc_info=True)
            console.print(f"[bold red]Error:[/bold red] {exc}")


if __name__ == "__main__":
    run()
