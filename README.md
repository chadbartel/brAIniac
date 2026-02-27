# brAIniac

A decentralized, local-first, uncensored AI assistant designed to run entirely on consumer hardware with an 8 GB VRAM ceiling (NVIDIA RTX 2070 SUPER). No cloud, no API keys, no telemetry — every inference happens on your machine.

## Goals

- **Private by design** — the LLM, tool execution, and conversation history never leave your machine.
- **Uncensored** — 4-bit quantized open-weight models via [Ollama](https://ollama.com); no content filters imposed upstream.
- **VRAM-disciplined** — a hard 8 GB ceiling is enforced at every layer: model selection, quantization, and a rolling context window that prevents memory bloat.
- **Modular and extensible** — tools are isolated FastMCP servers that can be added, replaced, or disabled without touching the core chat loop.

## Architecture

```text
brAIniac/
├── core/                        # Orchestration and rolling memory
│   ├── chat.py                  # Chat loop — Ollama integration, tool dispatch
│   └── memory.py                # Rolling context window (VRAM guard)
├── servers/                     # Isolated FastMCP tool servers
│   └── base_tools/
│       └── server.py            # get_current_time, web_search (DuckDuckGo)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml       # Ollama + web UI services
├── tests/                       # Full unit + integration test suite
├── main.py                      # Rich CLI entry point
├── web.py                       # Gradio web UI entry point
└── pyproject.toml               # Single root dependency manifest (Poetry)
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.12+ |
| Dependencies | Poetry (single root `pyproject.toml`) |
| LLM runtime | Ollama (Docker) |
| Default model | `llama3.1:8b-instruct-q4_K_M` (4-bit GGUF, ~5 GB VRAM) |
| Tool protocol | Model Context Protocol — FastMCP v2 |
| Web UI | Gradio 6 |
| CLI | Rich |
| Containers | Docker Compose |

## Hardware Requirements

- **GPU:** NVIDIA GPU with ≥ 8 GB VRAM (tested on RTX 2070 SUPER)
- **RAM:** 32 GB recommended (16 GB minimum)
- **Disk:** ~10 GB for model weights + dependencies
- **OS:** Linux (Docker, recommended) or Windows with WSL2

## Local Setup

### Prerequisites

1. **[Ollama](https://ollama.com/download)** installed and running — or use Docker Compose (see below).
2. **[Poetry](https://python-poetry.org/docs/#installation)** for dependency management.
3. NVIDIA drivers + [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) if running on GPU.

### 1. Clone and install

```bash
git clone <repo-url>
cd brAIniac
poetry install
```

### 2. Configure environment

Copy or create a `.env` file in the project root:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
MAX_CONTEXT_MESSAGES=20
```

### 3. Pull the model

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 4. Launch the web UI

```bash
poetry run python web.py
```

Then open **<http://localhost:7860>** in your browser.

The interface provides a streaming chat window, a clear-history button, and shows the active model and Ollama host in the footer.

### Alternative: CLI

```bash
poetry run python main.py
```

**CLI commands:**

| Command | Action |
| --- | --- |
| `/help` | Show available commands |
| `/clear` | Reset rolling memory |
| `/stats` | Show current context message count |
| `/quit` / `/exit` | Exit |

## Docker Setup (Full Stack)

Runs Ollama, model pull, and the Gradio web UI as a single Compose stack. The web UI is accessible at **<http://localhost>** (port 80 → container port 7860).

```bash
# From the project root
docker compose -f docker/docker-compose.yml up -d
```

Compose starts three services in order:

1. `ollama` — the LLM inference engine with GPU passthrough
2. `ollama-pull` — one-shot init container that pulls `$OLLAMA_MODEL`
3. `brainiac-web` — the Gradio web UI (waits for `ollama-pull` to complete)

To change the model, set `OLLAMA_MODEL` in your `.env` before bringing the stack up.

## VRAM Budget

Llama 3.1 8B at 4-bit quantization uses approximately 5–5.5 GB VRAM, leaving ~2.5 GB of headroom on an 8 GB card. brAIniac enforces this budget through:

- **Model selection** — no model larger than 8 B parameters is supported or recommended.
- **Rolling context window** — the last `MAX_CONTEXT_MESSAGES` (default 20) messages are kept; older turns are silently dropped.
- **Ollama isolation** — all inference runs inside the Ollama container; the application containers are CPU-only.

## Development

### Running Tests

```bash
# Full suite with coverage (61 tests, 100% coverage on core + servers)
poetry run pytest

# Single module
poetry run pytest tests/test_memory.py
poetry run pytest tests/test_chat.py
poetry run pytest tests/test_base_tools.py
poetry run pytest tests/test_integration.py

# HTML coverage report
poetry run pytest --cov-report=html
# → open htmlcov/index.html
```

Live integration tests (require a running Ollama instance) are excluded from the standard run and live in `tests/live_harness.py` and `tests/live_tool_isolation.py`:

```bash
python run_harness.py
python tests/live_tool_isolation.py
```

### Adding a New MCP Tool

1. Add a function decorated with `@mcp.tool()` in `servers/base_tools/server.py` (or create a new server module under `servers/`).
2. Follow the Google-style docstring convention — the docstring is what the LLM reads to decide when to call the tool.
3. Add coverage in `tests/test_base_tools.py`; mock any external I/O.

```python
@mcp.tool()
def my_tool(param: str) -> str:
    """One-line summary used by the LLM to decide when to call this.

    Args:
        param: What the parameter means.

    Returns:
        JSON-encoded result dict.
    """
    return json.dumps({"result": param})
```

### Code Quality

```bash
poetry run black .          # format
poetry run ruff check .     # lint
poetry run mypy .           # type-check
```

## Roadmap

### Phase 1 — Foundation ✅

- [x] Rolling memory buffer with FIFO eviction
- [x] Ollama integration (4-bit GGUF models)
- [x] FastMCP tools: `get_current_time`, `web_search` (DuckDuckGo, no API key)
- [x] Gradio web UI (`web.py`) — accessible at `http://localhost:7860`
- [x] Rich CLI (`main.py`)
- [x] Docker Compose stack (Ollama + web service)
- [x] 61 unit + integration tests, 100% coverage on core and servers

### Phase 2 — Advanced Context & Research

- [ ] `brainiac-research-server` with local SearXNG container and IterDRAG methodology
- [ ] Letta (MemGPT) replacing the rolling window for OS-level virtual context paging
- [ ] Full FastMCP client integration (live tool calling from the chat loop)

### Phase 3 — Voice & Multi-Agent Routing

- [ ] Ultra-low-latency STT (Canary 2.5B or Parakeet V3, CPU offloaded)
- [ ] Kokoro-82M TTS
- [ ] Intent router (Agent Squad or Observer framework) for multi-persona dispatch

### Phase 4 — Autonomous Self-Learning

- [ ] Nightly QLoRA fine-tuning with Unsloth on curated conversation history
- [ ] Hot-swappable LoRA adapter loading without service restart

## Troubleshooting

### `Connection refused` on Ollama

```bash
# Check the container
docker ps | grep ollama
docker logs brainiac-ollama

# Or verify the local daemon
curl http://localhost:11434/api/tags
```

### Model not found

```bash
# Pull manually
ollama pull llama3.1:8b-instruct-q4_K_M
# Or inside Docker
docker exec -it brainiac-ollama ollama pull llama3.1:8b-instruct-q4_K_M
```

### CUDA / GPU not detected

```bash
# Confirm NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check GPU inside the Ollama container
docker exec -it brainiac-ollama nvidia-smi
```

## License

[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
