"""
brainiac/servers/system-server/src/system_server/orchestrator.py

Two-layer heterogeneous orchestrator.

Architecture:
  - Layer 1: ToolRouter (small, fast model — e.g. llama3.2:3b)
      Analyses the user prompt and conversation history.
      Outputs a single structured JSON routing decision:
          {"required_collections": [], "web_search_needed": false, "query": ""}
      Does NOT generate prose.  It is a pure intent classifier + routing controller.

      Mode determination from router output:
        CHAT mode     — web_search_needed=False AND required_collections=[].
                        Skips Semantic Gap check, VDB query, and web search entirely.
                        Fast path directly to the Generator.
        RESEARCH mode — web_search_needed=True OR required_collections non-empty.
                        Runs Semantic Gap check; uses VDB context if coverage is
                        sufficient, otherwise executes live web search and delegates
                        results to the LibrarianAgent for VDB persistence.

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

# ---------------------------------------------------------------------------
# Router prompt builder — injects the live Library Card for Recursive Routing
# ---------------------------------------------------------------------------

_ROUTER_SCHEMA_DESCRIPTION: str = (
    '{"required_collections": ["<name>"], "web_search_needed": true|false, "query": "<string>"}'
)


def _build_router_prompt(library_card_json: str = "[]") -> str:
    """Build the Layer 1 router system prompt, injecting the live Library Card.

    The Library Card is a JSON list of existing VDB collections (name + summary)
    fetched from the research-server's Master Registry at the start of each
    run_task() call.  The router uses it to choose which existing collections
    are relevant before deciding whether live web search is also needed.

    Args:
        library_card_json: JSON string of collection descriptors.  Defaults to
            an empty list (no VDB context available yet).

    Returns:
        The complete system prompt string for the router model.
    """
    return (
        "You are a tool-routing controller. "
        "Your ONLY job is to analyse the user's message and output a routing "
        "decision.  Do NOT answer the question yourself.\n\n"
        "Output EXACTLY one JSON object and nothing else — no prose, no markdown, "
        "no code fences. The object MUST conform to this schema:\n\n"
        f"{_ROUTER_SCHEMA_DESCRIPTION}\n\n"
        "Field rules:\n"
        "  required_collections  — list of collection names from the Library Card "
        "(below) that are directly relevant to the question.  Use [] if none match.\n"
        "  web_search_needed     — true if the question asks for current/live/real-time "
        "information (weather, news, prices, scores, events) or uses words like "
        "'now', 'today', 'latest', 'current'. false for greetings, maths, code, "
        "opinions, creative writing, or historical/static facts.\n"
        "  query                 — a concise search / retrieval query for the question.\n\n"
        "--- LIBRARY CARD (available VDB collections) ---\n"
        f"{library_card_json}\n"
        "--- END LIBRARY CARD ---\n\n"
        "If the Library Card is empty or no collection is relevant, set "
        "required_collections to [].\n"
        'Fallback (no tools, no VDB): {"required_collections": [], "web_search_needed": false, "query": ""}'
    )


@dataclasses.dataclass(slots=True)
class ToolRouterResult:
    """Structured output from the tool-routing layer (Layer 1).

    Attributes:
        required_collections: VDB collection names the router flagged as relevant.
        web_search_needed: Whether the router determined a live web search is required.
        query: The retrieval / search query string.
        raw: The raw JSON string returned by the router model.
    """

    required_collections: list[str]
    web_search_needed: bool
    query: str
    raw: str


def _call_tool_router(
    user_prompt: str,
    history: list[dict[str, Any]] | None = None,
    *,
    router_model: str | None = None,
    library_card_json: str = "[]",
) -> ToolRouterResult:
    """Call the router model (Layer 1) via the Ollama chat-completions API.

    Uses a direct ``httpx`` call rather than AutoGen to avoid GroupChat
    overhead and to enforce strict JSON-only output from the small model.
    History is passed as proper ``messages`` array entries so the model
    receives a structured multi-turn conversation rather than a text blob.
    The live Library Card is injected into the system prompt so the router
    can reference existing VDB collections for Recursive Routing (§8, Task 3).
    On any parse failure the function returns a safe no-tool result so the
    pipeline never blocks.

    Args:
        user_prompt: The current user request (not including history).
        history: Prior conversation turns as
            ``[{"role": "user"|"assistant", "content": "..."}]``.
        router_model: Override the router model for this call.  Defaults to
            the ``OLLAMA_MODEL_ROUTER`` setting.
        library_card_json: JSON string of Master Registry collection descriptors
            fetched at the start of the current ``run_task()`` call.

    Returns:
        A :class:`ToolRouterResult` with collection names, web-search flag, and query.
    """
    model: str = router_model or _ROUTER_MODEL
    completions_url: str = cfg.ollama_base_url.rstrip("/") + "/chat/completions"

    system_prompt: str = _build_router_prompt(library_card_json)

    # Build messages: system prompt, then history turns, then current question.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
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

        # Strip Markdown code fences small models sometimes wrap JSON in.
        clean_text = re.sub(r"^```[a-z]*\n?", "", raw_text, flags=re.IGNORECASE)
        clean_text = re.sub(r"\n?```$", "", clean_text).strip()

        parsed: dict[str, Any] = json.loads(clean_text)

        # ----------------------------------------------------------------
        # Support both new schema and legacy schema from older model output.
        # New: {required_collections, web_search_needed, query}
        # Legacy: {tools_needed, queries}  — map to new fields automatically.
        # ----------------------------------------------------------------
        if "required_collections" in parsed or "web_search_needed" in parsed:
            required_collections: list[str] = list(
                parsed.get("required_collections") or []
            )
            web_search_needed: bool = bool(parsed.get("web_search_needed", False))
            query: str = str(parsed.get("query") or "")
        else:
            # Legacy fallback: map tools_needed=["search_web"] → web_search_needed=True
            legacy_tools: list[str] = list(parsed.get("tools_needed") or [])
            legacy_queries: list[str] = list(parsed.get("queries") or [])
            web_search_needed = "search_web" in legacy_tools
            query = legacy_queries[0] if legacy_queries else ""
            required_collections = []

        result = ToolRouterResult(
            required_collections=required_collections,
            web_search_needed=web_search_needed,
            query=query,
            raw=raw_text,
        )
        logger.info(
            "[tool_router] required_collections=%s web_search_needed=%s query=%r",
            result.required_collections,
            result.web_search_needed,
            result.query,
        )
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
    return ToolRouterResult(
        required_collections=[], web_search_needed=False, query="", raw=""
    )


# ---------------------------------------------------------------------------
# Library Card fetch — populates the router's Library Card context
# ---------------------------------------------------------------------------


def _fetch_library_card() -> str:
    """Fetch the Master Registry library card from the research-server.

    Returns a JSON list of ``{collection_name, summary, last_updated, doc_count}``
    dicts for all registered leaf VDB collections.  This list is injected into
    the router's system prompt every turn so the 3B model can perform Recursive
    Routing (§8, Task 3).

    On any network or parse error the function returns an empty JSON array so
    the pipeline degrades gracefully to web-search-only routing.

    Returns:
        JSON-encoded list string, e.g. ``'[{"collection_name": "python_abc", ...}]'``.
    """
    try:
        result_json: str = _call_research_tool("get_library_card")
        # Validate it parses cleanly; fall back to empty list on failure.
        parsed = json.loads(result_json)
        if isinstance(parsed, list):
            logger.info("[library_card] fetched %d collection entries", len(parsed))
            return result_json
        logger.warning(
            "[library_card] unexpected response type %s — using empty card",
            type(parsed).__name__,
        )
    except Exception as exc:
        logger.warning(
            "[library_card] could not fetch library card: %s", exc
        )
    return "[]"


# ---------------------------------------------------------------------------
# Semantic Gap check — determines if VDB context is sufficient
# ---------------------------------------------------------------------------


def _check_semantic_gap(query: str) -> dict[str, Any]:
    """Query the Master Registry and evaluate whether a Semantic Gap exists.

    A Semantic Gap is declared when the best-matching VDB collection summary
    scores below SEMANTIC_GAP_THRESHOLD (0.85) against the user query.

    Args:
        query: The user's natural-language question or retrieval query.

    Returns:
        Parsed dict from the research-server ``query_master_registry`` tool:
        ``{max_similarity, has_gap, semantic_gap_threshold, collections}``.
        Returns a synthetic gap-detected dict on any error.
    """
    try:
        result_json: str = _call_research_tool(
            "query_master_registry", query=query, top_k=3
        )
        result: Any = json.loads(result_json)
        if isinstance(result, dict) and "has_gap" in result:
            logger.info(
                "[semantic_gap] max_similarity=%.4f has_gap=%s",
                result.get("max_similarity", 0.0),
                result.get("has_gap", True),
            )
            return result
    except Exception as exc:
        logger.warning("[semantic_gap] check failed: %s", exc)
    # Fail-safe: assume gap detected so we always fall back to web search.
    return {"max_similarity": 0.0, "has_gap": True, "collections": []}


# ---------------------------------------------------------------------------
# Librarian delegation — persist research results into the VDB
# ---------------------------------------------------------------------------


def _delegate_to_librarian(
    web_results_json: str,
    query: str,
    gap_check: dict[str, Any],
) -> None:
    """Persist web search results into the VDB via the LibrarianAgent tools.

    Implements the Research → Vectorize → Register workflow (§8.2):
      1. Convert each web-search hit into a ``{text, metadata}`` document.
      2. Compare ``max_similarity`` from the Semantic Gap check against
         ``UPDATE_THRESHOLD`` (0.95) to decide create vs. update.
      3. Call the appropriate research-server tool asynchronously so the
         response latency is not added to the user-facing answer time.

    Args:
        web_results_json: JSON string of ``[{title, url, snippet}]`` dicts.
        query: The original retrieval query (used as topic summary hint).
        gap_check: Parsed result from :func:`_check_semantic_gap`.
    """
    try:
        raw_results: Any = json.loads(web_results_json)
        if not isinstance(raw_results, list) or not raw_results:
            return

        # Convert web hits to VDB document schema.
        documents: list[dict[str, Any]] = [
            {
                "text": (
                    f"{hit.get('title', '')}\n{hit.get('snippet', '')}"
                ).strip(),
                "metadata": {
                    "source_url": hit.get("url", ""),
                    "query": query,
                    "related_collections": [],
                },
            }
            for hit in raw_results
            if hit.get("snippet") or hit.get("title")
        ]
        if not documents:
            return

        max_similarity: float = float(gap_check.get("max_similarity", 0.0))
        collections: list[dict[str, Any]] = gap_check.get("collections", [])
        best_collection: str = (
            collections[0].get("collection_name", "") if collections else ""
        )

        # UPDATE_THRESHOLD from the research-server librarian (§8.3).
        UPDATE_THRESHOLD_LOCAL: float = 0.95

        if max_similarity >= UPDATE_THRESHOLD_LOCAL and best_collection:
            logger.info(
                "[librarian] similarity=%.4f ≥ %.2f — updating collection %r",
                max_similarity, UPDATE_THRESHOLD_LOCAL, best_collection,
            )
            _call_research_tool(
                "update_collection",
                name=best_collection,
                documents=documents,
                summary="",
            )
        else:
            logger.info(
                "[librarian] similarity=%.4f < %.2f — creating new collection for query %r",
                max_similarity, UPDATE_THRESHOLD_LOCAL, query,
            )
            _call_research_tool(
                "create_collection",
                name=re.sub(r"\W+", "_", query[:40].lower()).strip("_"),
                summary=query,
                documents=documents,
            )
    except Exception as exc:
        logger.error("[librarian] delegation failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_web_search(query: str) -> list[dict[str, Any]]:
    """Execute a web search and return structured tool output entries.

    Replaces the old ``_execute_tools`` for the single search_web case.
    Error results are silently dropped so the generator never receives
    error JSON as fake research data.

    Args:
        query: The search query string.

    Returns:
        A list with zero or one ``{tool, query, result}`` dict.
    """
    if not query:
        return []

    logger.info("[execute_web_search] query=%r", query)
    result_json: str = _search_web_direct(query)

    try:
        parsed = json.loads(result_json)
        if isinstance(parsed, dict) and "error" in parsed:
            logger.warning(
                "[execute_web_search] search returned error — skipping: %s",
                parsed["error"],
            )
            return []
    except json.JSONDecodeError:
        pass

    logger.info("[execute_web_search] returned %d chars", len(result_json))
    return [{"tool": "search_web", "query": query, "result": result_json}]


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
      1. Human approval gate — keyword check against the prompt.
      2. Fetch Library Card from Master Registry + Layer 1 ToolRouter:
         structured JSON routing decision (required_collections, web_search_needed, query).
      3. Semantic Gap check — query Master Registry to measure VDB coverage.
         a. If gap < SEMANTIC_GAP_THRESHOLD and VDB has relevant collections:
            inject VDB context, skip web search.
         b. If gap detected or router requested web search: run web search,
            then delegate results to LibrarianAgent (create or update collection).
      4. Layer 2 — Generator: synthesises final answer, optionally grounded
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
    # Step 2 — Fetch Library Card and run Layer 1 (ToolRouter)
    # The live Library Card is injected into the router's system prompt for
    # Recursive Routing so the 3B model knows which VDB collections exist.
    # -----------------------------------------------------------------------
    _emit(
        on_message,
        agent="Librarian",
        recipient="Orchestrator",
        content="[Library Card] Fetching Master Registry catalogue...",
        event_type="message",
    )
    library_card_json: str = _fetch_library_card()
    _emit(
        on_message,
        agent="Librarian",
        recipient="Orchestrator",
        content=f"[Library Card] {library_card_json[:200]}{'...' if len(library_card_json) > 200 else ''}",
        event_type="message",
    )

    _emit(
        on_message,
        agent="ToolRouter",
        recipient="Orchestrator",
        content=f"[Routing] Analysing prompt with {router_model or _ROUTER_MODEL}...",
        event_type="message",
    )
    router_result: ToolRouterResult = _call_tool_router(
        user_prompt,
        history,
        router_model=router_model,
        library_card_json=library_card_json,
    )
    routing_summary: str = (
        f"[ToolRouter] required_collections={router_result.required_collections}  "
        f"web_search_needed={router_result.web_search_needed}  "
        f"query={router_result.query!r}"
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
    # Step 3 — Semantic Gap check + tool execution + Librarian delegation
    #
    # Mode determination (derived from Layer 1 router decision):
    #   CHAT mode   — router returned web_search_needed=False AND
    #                 required_collections=[].  No VDB or web tooling is
    #                 needed; skip the Semantic Gap check entirely and go
    #                 straight to the Generator.
    #   RESEARCH mode — router flagged web_search_needed=True OR named at
    #                 least one VDB collection.  Run the Semantic Gap check
    #                 to decide whether VDB coverage is sufficient or a live
    #                 web search is required.
    #
    # The ToolRouter IS the intent classifier.  The Semantic Gap check must
    # only run once the router has already decided research *could* be
    # relevant — running it unconditionally for every query (including
    # greetings) allows a 0.0 similarity score to override the router's
    # legitimate "no tools needed" decision.
    # -----------------------------------------------------------------------
    tool_outputs: list[dict[str, Any]] = []

    # Derive execution mode from the router's structured output.
    needs_research: bool = router_result.web_search_needed or bool(
        router_result.required_collections
    )
    search_query: str = router_result.query or user_prompt

    if not needs_research:
        # ------------------------------------------------------------------
        # CHAT mode — router says no tools needed.  Skip all research
        # pipeline stages (Semantic Gap, web search, Librarian delegation).
        # ------------------------------------------------------------------
        logger.info(
            "[orchestrator] CHAT mode — router returned no tools needed; "
            "skipping Semantic Gap check and web search."
        )
        gap_check: dict[str, Any] = {
            "max_similarity": 1.0,
            "has_gap": False,
            "collections": [],
        }
        has_gap: bool = False

    else:
        # ------------------------------------------------------------------
        # RESEARCH mode — run Semantic Gap check to decide VDB vs web search.
        # ------------------------------------------------------------------
        gap_check = _check_semantic_gap(search_query)
        has_gap = bool(gap_check.get("has_gap", True))
        max_sim: float = float(gap_check.get("max_similarity", 0.0))
        _emit(
            on_message,
            agent="Librarian",
            recipient="Orchestrator",
            content=(
                f"[SemanticGap] max_similarity={max_sim:.4f} "
                f"has_gap={has_gap}  "
                f"threshold={gap_check.get('semantic_gap_threshold', 0.85)}"
            ),
            event_type="message",
        )

        # If the VDB has adequate coverage, retrieve context from it directly.
        vdb_collections: list[dict[str, Any]] = gap_check.get("collections", [])
        if not has_gap and vdb_collections:
            logger.info(
                "[orchestrator] VDB context sufficient (max_similarity=%.4f) "
                "— skipping web search.",
                max_sim,
            )
            _emit(
                on_message,
                agent="Librarian",
                recipient="Orchestrator",
                content=(
                    f"[VDB Hit] Answering from memory "
                    f"(collections: {[c['collection_name'] for c in vdb_collections]})"
                ),
                event_type="function_call",
            )

        # Web search: triggered when router requested it OR a Semantic Gap exists.
        if router_result.web_search_needed or has_gap:
            if not search_query:
                logger.warning(
                    "[orchestrator] web search flagged but query is empty — skipping"
                )
            else:
                tool_outputs = _execute_web_search(search_query)
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
                # Delegate to LibrarianAgent when a Semantic Gap triggered the search.
                # Persistence is driven by gap detection, not by query type.
                # If has_gap=True, the VDB lacks this knowledge → learn it.
                # If has_gap=False, VDB coverage was sufficient → nothing to store.
                if tool_outputs and has_gap:
                    _emit(
                        on_message,
                        agent="Librarian",
                        recipient="Orchestrator",
                        content="[Librarian] Persisting research results to VDB...",
                        event_type="message",
                    )
                    _delegate_to_librarian(
                        tool_outputs[0]["result"],
                        search_query,
                        gap_check,
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
