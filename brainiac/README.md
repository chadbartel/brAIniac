# brAIniac

> Decentralized, local-first AI ecosystem — modular, containerized, uncensored.

## Architecture

```
brainiac/
├── servers/
│   ├── research-server/        # FastMCP: DuckDuckGo search + ChromaDB memory
│   ├── system-server/          # AutoGen orchestrator + dynamic agent delegation
│   └── voice-server/           # faster-whisper STT + Kokoro/Piper TTS (optional)
├── docker/
│   ├── docker-compose.yml      # Full stack orchestration
│   ├── Dockerfile.research
│   ├── Dockerfile.system
│   ├── Dockerfile.voice
│   └── tailscale/              # Secure remote access via Tailscale
│       └── tailscaled.env
└── pyproject.toml              # Workspace-level dev tooling
```

### Core Design Principles

| Principle | Implementation |
|---|---|
| **Local-first** | All inference via Ollama (GGUF models, no cloud calls) |
| **Dynamic delegation** | OrchestratorAgent decomposes tasks → instantiates worker agents per step |
| **Single Responsibility** | Each MCP server owns exactly one domain |
| **Human-in-the-loop** | Destructive actions gate on explicit human approval |
| **Uncensored** | `dolphin-llama3` (or any uncensored GGUF) via Ollama |

---

## Quick Start

### Prerequisites

- Docker Engine ≥ 26 with the Compose plugin
- (Optional) NVIDIA GPU + `nvidia-container-toolkit` for GPU inference
- A Tailscale auth key (only if you want remote access)

### 1. Configure environment

```bash
cp docker/.env.example docker/.env
# Edit docker/.env — set TAILSCALE_AUTHKEY, OLLAMA_MODEL, etc.
```

### 2. Start the stack

```bash
cd docker
docker compose up -d
```

Ollama will automatically pull `dolphin-llama3` on first start (≈ 5 GB).

To include the voice server:

```bash
docker compose --profile voice up -d
```

### 3. Run the orchestrator interactively

```bash
docker compose exec system-server system-server
```

---

## Services & Ports

| Service | Port | Description |
|---|---|---|
| Ollama | 11434 | OpenAI-compatible LLM inference |
| ChromaDB | 8000 | Vector memory backend |
| research-server | 8100 | FastMCP: search + memory tools |
| system-server | 8300 | AutoGen orchestrator REPL |
| voice-server | 8200 | STT/TTS (opt-in via `--profile voice`) |

---

## Server Details

### research-server

FastMCP server exposing three tools over SSE (`http://research-server:8100/sse`):

| Tool | Description |
|---|---|
| `search_web(query, max_results)` | DuckDuckGo text search |
| `store_memory(text, metadata, doc_id)` | Embed and persist to ChromaDB |
| `query_memory(query, top_k)` | Semantic recall from ChromaDB |

### system-server (Orchestrator)

AutoGen `GroupChat` with three agents:

- **OrchestratorAgent** — decomposes tasks, builds plans, delegates, synthesises results
- **ResearchAgent** — calls research-server tools; checks memory before searching
- **HumanProxy** — intercepts any action containing sensitive keywords (`delete`, `deploy`, etc.) and requires explicit `yes` before proceeding

### voice-server

Scaffold for STT (`faster-whisper`) and TTS (Kokoro/Piper). Enable the TTS
backend of your choice by uncommenting the relevant lines in
`servers/voice-server/pyproject.toml` and `server.py`.

---

## Development

Each server is an independent Poetry project. To work on one locally:

```bash
cd servers/research-server
poetry install
poetry run research-server
```

### Code Quality

```bash
# From any server directory:
poetry run black src/
poetry run isort src/
poetry run flake8 src/
poetry run mypy src/
poetry run pytest
```

---

## Extending the System

1. **Add a new MCP server** — create `servers/<name>-server/` following the
   research-server pattern. Expose tools via `@mcp.tool()`.
2. **Register a new worker agent** — add a factory function in
   `system-server/src/system_server/orchestrator.py`, create the agent with
   `llm_config` pointing at Ollama, and register it in the `GroupChat`.
3. **Swap the LLM** — set `OLLAMA_MODEL` in `.env` to any model tag available
   on Ollama Hub (e.g. `llama3:70b-instruct`, `mistral`, `qwen2`).

---

## License

See [LICENSE](../LICENSE).
