# brAIniac - Phase 1

Local-first, uncensored AI chatbot optimized for 8GB VRAM (RTX 2070 SUPER).

## Architecture Overview

brAIniac is built with strict modularity and Single Responsibility Principle in mind:

```
brAIniac/
├── core/                  # Orchestration, rolling memory, and agent routing
│   ├── __init__.py
│   ├── chat.py           # Main chat loop with Ollama integration
│   └── memory.py         # Rolling context window (prevents VRAM exhaustion)
├── servers/               # Isolated MCP servers
│   └── base_tools/       # FastMCP server (time, web search)
│       ├── __init__.py
│       └── server.py
├── docker/                # Dockerfiles and compose configs
│   ├── Dockerfile
│   └── docker-compose.yml
├── main.py                # CLI entry point
└── pyproject.toml         # Poetry dependencies
```

## Tech Stack

- **Language:** Python 3.12+
- **Dependency Management:** Poetry
- **Containerization:** Docker & Docker Compose
- **LLM Orchestration:** Ollama (local)
- **Model:** Llama 3.1 8B Instruct (4-bit GGUF) or Qwen 2.5 7B (4-bit GGUF)
- **Tooling Protocol:** Model Context Protocol (MCP) using FastMCP
- **CLI:** Rich (formatted terminal output)

## Hardware Requirements

- **GPU:** NVIDIA RTX 2070 SUPER (8GB VRAM) or equivalent
- **RAM:** 16GB+ recommended
- **Disk:** 10GB+ for models and dependencies

## Quick Start

### Prerequisites

1. **NVIDIA GPU Drivers** and **NVIDIA Container Toolkit** installed
2. **Docker** and **Docker Compose** installed
3. **Poetry** installed (for local development)

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd brAIniac

# Start the services (Ollama + brAIniac)
docker compose -f docker/docker-compose.yml up -d

# Pull the model (first time only)
docker exec -it brainiac-ollama ollama pull llama3.1:8b-instruct-q4_K_M

# Run the interactive CLI
docker attach brainiac-app
```

### Option 2: Local Development

```bash
# Install dependencies
poetry install

# Start Ollama separately (ensure it's running on localhost:11434)
# Then run the CLI
poetry run python main.py
```

### Configuration

Create a `.env` file in the project root:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
MAX_CONTEXT_MESSAGES=20
```

## Usage

### Interactive Commands

- `/help` - Show available commands
- `/clear` - Clear conversation history (reset rolling memory)
- `/stats` - Show context statistics (message count, VRAM usage)
- `/quit` or `/exit` - Exit brAIniac

### Example Session

```
You: What's the current time?

brAIniac: [Uses get_current_time tool and responds]

You: Search for Python best practices

brAIniac: [Uses web_search tool (mock in Phase 1)]

You: /stats

[Shows: Messages in context: 4/20, Context utilization: 20.0%]
```

## VRAM Protection Strategy

brAIniac protects your 8GB VRAM ceiling through:

1. **4-bit Quantized Models** - Llama 3.1 8B at 4-bit uses ~4.5-5.5GB VRAM
2. **Rolling Context Window** - Maximum 20 messages prevents context bloat
3. **Ollama Isolation** - All LLM inference delegated to separate Docker container
4. **Lazy Tool Loading** - MCP tools loaded only when needed

This leaves 2.5-3.5GB VRAM headroom for future additions (TTS, LoRA tuning).

## Roadmap

### Phase 1 (Current) ✅
- [x] Rolling memory buffer
- [x] Ollama integration
- [x] Mock FastMCP tools (time, web search)
- [x] Docker deployment
- [x] Rich CLI interface

### Phase 2 (Planned)
- [ ] Replace mock web_search with SearXNG container
- [ ] Full FastMCP client integration (actual tool calling)
- [ ] Add ChromaDB for semantic memory
- [ ] Implement research-server with IterDRAG

### Phase 3 (Future)
- [ ] Kokoro TTS integration
- [ ] Letta (MemGPT) for virtual context management
- [ ] Multi-agent orchestration
- [ ] Web UI (Gradio or FastAPI + HTMX)

## Development

### Adding New MCP Tools

1. Create a new tool in `servers/base_tools/server.py`:

```python
@mcp.tool()
def my_new_tool(param: str) -> str:
    """Tool description.

    Args:
        param: Parameter description.

    Returns:
        JSON-encoded result.
    """
    # Implementation
    return json.dumps({"result": "value"})
```

2. Update `core/chat.py` to handle the new tool in `execute_tool()`.

### Running Tests

```bash
poetry run pytest
```

### Code Quality

```bash
# Format code
poetry run black .

# Lint
poetry run ruff check .

# Type check
poetry run mypy .
```

## Troubleshooting

### Ollama Connection Refused

```bash
# Check if Ollama is running
docker ps | grep ollama

# View Ollama logs
docker logs brainiac-ollama
```

### Model Not Found

```bash
# Pull the model manually
docker exec -it brainiac-ollama ollama pull llama3.1:8b-instruct-q4_K_M

# Or use alternative model
docker exec -it brainiac-ollama ollama pull qwen2.5:7b-instruct-q4_K_M
```

### CUDA/GPU Issues

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check GPU visibility in Ollama container
docker exec -it brainiac-ollama nvidia-smi
```

## License

[Your chosen license]

## Contributing

Contributions welcome! Please follow the existing architecture patterns and ensure all code is fully typed (Python 3.12+ type hints).
