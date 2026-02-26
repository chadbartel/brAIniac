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
import urllib.parse
import urllib.request
from typing import Any, Generator
from datetime import datetime

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
            "name": "get_weather",
            "description": (
                "Get the current weather and multi-day forecast for a location. "
                "Use this for ANY question about weather conditions, temperature, "
                "rain, wind, or forecasts — current or future."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or location name (e.g. 'Seattle', 'London', 'New York').",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of forecast days to return: 1 (today only), 2 (today + tomorrow), or 3 (3-day). Default 2.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for factual, real-time, or current information. "
                "Use for news, prices, sports scores, recent events, coding questions, "
                "and general knowledge. Do NOT use for weather — use get_weather instead. "
                "Do NOT use for greetings, casual conversation, or creative writing."
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
        arguments: Parsed arguments dict from the model's tool call.

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

    if name == "get_weather":
        location: str = str(arguments.get("location", "Seattle"))
        days_raw = arguments.get("days", 2)
        if isinstance(days_raw, dict):
            days_raw = 2
        days: int = max(1, min(3, int(days_raw)))

        # WMO weather interpretation codes → human-readable condition string.
        _WMO: dict[int, str] = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Icy fog",
            51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
            80: "Light showers", 81: "Showers", 82: "Heavy showers",
            85: "Snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail",
        }

        def _wmo_desc(code: int) -> str:
            return _WMO.get(code, f"WMO code {code}")

        def _deg_to_compass(deg: float) -> str:
            dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                    "S","SSW","SW","WSW","W","WNW","NW","NNW"]
            return dirs[round(deg / 22.5) % 16]

        def _fetch(url: str) -> dict[str, Any]:
            req = urllib.request.Request(url, headers={"User-Agent": "brAIniac/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())  # type: ignore[return-value]

        try:
            # Step 1 — geocode the city name to lat/lon via Open-Meteo's
            # geocoding API (free, no key required).
            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search?"
                + urllib.parse.urlencode({"name": location, "count": 1, "format": "json"})
            )
            geo = _fetch(geo_url)
            if not geo.get("results"):
                return json.dumps({"error": f"Location not found: {location!r}"})
            hit = geo["results"][0]
            lat: float = hit["latitude"]
            lon: float = hit["longitude"]
            resolved_name: str = hit.get("name", location)
            country: str = hit.get("country", "")

            # Step 2 — fetch current conditions + daily forecast via
            # Open-Meteo forecast API (free, no key required).
            forecast_url = (
                "https://api.open-meteo.com/v1/forecast?"
                + urllib.parse.urlencode({
                    "latitude": lat,
                    "longitude": lon,
                    "current": ",".join([
                        "temperature_2m", "apparent_temperature", "weathercode",
                        "windspeed_10m", "winddirection_10m", "relativehumidity_2m",
                    ]),
                    "daily": ",".join([
                        "temperature_2m_max", "temperature_2m_min",
                        "weathercode", "precipitation_sum",
                    ]),
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "precipitation_unit": "inch",
                    "timezone": "auto",
                    "forecast_days": days,
                })
            )
            wx = _fetch(forecast_url)
            cur = wx["current"]
            daily = wx["daily"]
            day_labels = ["today", "tomorrow", "day_after_tomorrow"]
            result: dict[str, Any] = {
                "location": f"{resolved_name}, {country}".strip(", "),
                "current": {
                    "temp_f": round(cur["temperature_2m"]),
                    "feels_like_f": round(cur["apparent_temperature"]),
                    "condition": _wmo_desc(int(cur["weathercode"])),
                    "humidity_pct": int(cur["relativehumidity_2m"]),
                    "wind_mph": round(cur["windspeed_10m"]),
                    "wind_dir": _deg_to_compass(float(cur["winddirection_10m"])),
                },
                "forecast": [
                    {
                        "day": day_labels[i] if i < len(day_labels) else f"day_{i}",
                        "date": daily["time"][i],
                        "high_f": round(daily["temperature_2m_max"][i]),
                        "low_f": round(daily["temperature_2m_min"][i]),
                        "condition": _wmo_desc(int(daily["weathercode"][i])),
                        "precip_in": round(daily["precipitation_sum"][i], 2),
                    }
                    for i in range(len(daily["time"]))
                ],
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            logger.error("[get_weather] Open-Meteo failure: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc)})

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
    return " ".join(part.get("text", "") for part in content if isinstance(part, dict))


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

# Load .env before reading any os.environ values so that the project-root
# .env file is always the authoritative configuration source.
load_dotenv()

# Support both OLLAMA_HOST (.env convention) and OLLAMA_BASE_URL (legacy).
# OLLAMA_HOST takes priority; OLLAMA_BASE_URL is accepted as a fallback.
DEFAULT_OLLAMA_HOST: str = (
    os.environ.get("OLLAMA_HOST")
    or os.environ.get("OLLAMA_BASE_URL")
    or "http://localhost:11434"
)
DEFAULT_MODEL: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
DEFAULT_MAX_CTX: int = 20
HARNESS_PORT: int = int(os.environ.get("HARNESS_PORT", "7861"))

# Backoff delays (seconds) between successive Ollama retry attempts.
# Two retries = three total attempts: immediate → wait 1 s → wait 2 s → fail.
_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)

# System prompt sent as the first message in every chat context.
# Explicit tool-use rules prevent the model from answering real-time
# queries from stale training data instead of calling the provided tools.
SYSTEM_PROMPT: str = """\
You are brAIniac, a helpful local AI assistant. Respond conversationally and \
naturally. For greetings like "hello" or "how are you", reply warmly and briefly \
as yourself — do not reference songs, trivia, or anything other than the greeting.

Tool rules — follow these without exception:
1. ALWAYS call `get_current_time` when the user asks what time or date it is.
2. ALWAYS call `get_weather` for ANY question about weather — current conditions, \
forecasts, temperature, rain, wind, or whether to wear a jacket.
3. ALWAYS call `web_search` for news, sports scores, prices, or recent events.
4. NEVER answer questions requiring current data from your training knowledge.
5. For greetings, casual conversation, coding, or general knowledge that does NOT \
require up-to-date facts, answer directly — no tools needed.

When you receive <tool_results> tags in a message, synthesise your answer \
using ONLY those results. Do not contradict or ignore them."""

# ---------------------------------------------------------------------------
# Ollama retry helper
# ---------------------------------------------------------------------------


def _chat_with_retry(client: Any, **kwargs: Any) -> Any:
    """Call ``client.chat`` with exponential backoff on transient failures.

    Makes up to ``len(_RETRY_DELAYS) + 1`` total attempts.  Each failure
    (except the last) logs a warning and sleeps for the next value in
    ``_RETRY_DELAYS`` before retrying.  The final failure re-raises so the
    caller's error path handles it normally.

    Args:
        client: Ollama ``Client`` instance.
        **kwargs: Keyword arguments forwarded verbatim to ``client.chat``.

    Returns:
        The ``client.chat`` response object.

    Raises:
        Exception: Re-raises the last exception once all attempts are exhausted.
    """
    total_attempts: int = len(_RETRY_DELAYS) + 1
    last_exc: Exception
    for attempt in range(1, total_attempts + 1):
        try:
            return client.chat(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < total_attempts:
                delay: float = _RETRY_DELAYS[attempt - 1]
                logger.warning(
                    "[ollama] attempt %d/%d failed (%s) — retrying in %.1fs…",
                    attempt,
                    total_attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[ollama] all %d attempts exhausted: %s",
                    total_attempts,
                    exc,
                    exc_info=True,
                )
    raise last_exc  # type: ignore[possibly-undefined]


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
        # Filter out any engine-injected system messages so we only use the
        # single authoritative SYSTEM_PROMPT defined in this harness.
        _session.engine.memory.clear()
        for turn in history:
            role = turn.get("role", "")
            content = _str_content(turn.get("content", ""))
            if role in ("user", "assistant") and content:
                _session.engine.memory.add_message(role, content)

        # Build the full synthesis context (history + current user message).
        # Exclude the engine's internal system message (role="system") to
        # prevent a duplicate / conflicting system prompt in the context.
        _session.engine.memory.add_message("user", message)
        history_messages: list[dict[str, Any]] = [
            m
            for m in _session.engine.memory.get_context()
            if m.get("role") != "system"
        ]
        context_with_system: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history_messages,
        ]

        # ------------------------------------------------------------------
        # Pass 1 (non-streaming): TOOL SELECTION ONLY.
        #
        # CRITICAL: send ONLY [system, current_user_message] — no history.
        #
        # Prior turns whose assistant responses contain tool-result data
        # (e.g. weather about Seattle) contaminate the tool-selection signal
        # for subsequent unrelated questions (e.g. pygame code changes).
        # Isolating Pass 1 to the current message ensures tool selection is
        # driven solely by the user's *present* intent, not conversation
        # history artefacts.  Full history is restored for synthesis below.
        # ------------------------------------------------------------------
        pass1_context: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ]

        first_resp = _chat_with_retry(
            _session.engine.client,
            model=_session.model,
            messages=pass1_context,
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
                injected_parts.append(f"[{tc.function.name} result]\n{result_json}")

            injected_context = "\n\n".join(injected_parts)

            # Detect whether every tool call came back with an error dict so
            # we can give the model truthful synthesis instructions.  Telling
            # the model "the data has been retrieved" when the tool returned
            # {"error": ...} creates a conflicting instruction that causes it
            # to produce unhelpful "I'm unable to retrieve" apology responses.
            all_errors: bool = all(
                "error" in json.loads(p.split("\n", 1)[1])
                for p in injected_parts
                if "\n" in p
            )
            if all_errors:
                synthesis_instruction = (
                    f"<tool_results>\n{injected_context}\n</tool_results>\n\n"
                    f"The tool call above returned an error — the external "
                    f"service is temporarily unavailable. Inform the user "
                    f"concisely that the service is temporarily unavailable "
                    f"and suggest they try again in a moment. "
                    f"Do NOT say you lack access to real-time data.\n\n"
                    f"Question: {message}"
                )
            else:
                synthesis_instruction = (
                    f"<tool_results>\n{injected_context}\n</tool_results>\n\n"
                    f"Using ONLY the tool results above, answer the following "
                    f"question. Do not say you lack access to real-time data — "
                    f"the data has already been retrieved for you.\n\n"
                    f"Question: {message}"
                )

            # Replace the last user message in the full-history context with
            # the augmented one, then stream synthesis.  Full history is used
            # here (not pass1_context) so the model can produce a
            # contextually coherent response even for follow-up questions.
            synthesis_context = context_with_system[:-1] + [
                {"role": "user", "content": synthesis_instruction}
            ]

            # ------------------------------------------------------------------
            # Pass 2 (streaming): synthesis with tool results injected.
            # Retry the entire stream on failure so a transient Ollama hiccup
            # does not surface as a hard error to the user.
            # ------------------------------------------------------------------
            total_attempts: int = len(_RETRY_DELAYS) + 1
            last_stream_exc: Exception
            for _attempt in range(1, total_attempts + 1):
                try:
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
                    break  # stream completed successfully — exit retry loop
                except Exception as exc:
                    last_stream_exc = exc
                    if _attempt < total_attempts:
                        delay = _RETRY_DELAYS[_attempt - 1]
                        logger.warning(
                            "[ollama-stream] attempt %d/%d failed (%s) — retrying in %.1fs…",
                            _attempt,
                            total_attempts,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "[ollama-stream] all %d attempts exhausted: %s",
                            total_attempts,
                            exc,
                            exc_info=True,
                        )
                        raise last_stream_exc  # type: ignore[possibly-undefined]

        else:
            # ------------------------------------------------------------------
            # No tools needed — yield Pass 1's response directly.
            # Avoids a redundant second inference call for every simple reply.
            # ------------------------------------------------------------------
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
