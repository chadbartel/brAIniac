# brAIniac

## Project Overview

brAIniac is a local AI-powered assistant that leverages large language models (LLMs) to provide intelligent responses and assistance. The project uses llama.cpp with CUDA acceleration to run quantized models efficiently on local hardware, providing privacy and control over your AI interactions.

## Features

- Local LLM server using llama.cpp with CUDA support
- Docker containerization for easy deployment
- OpenAI-compatible API endpoint
- GPU acceleration for faster inference
- Support for various quantized GGUF model formats

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11 (WSL 2 not required)
- **GPU**: NVIDIA GPU with CUDA support (Compute Capability 7.0+)
- **RAM**: 16GB+ recommended
- **Storage**: Sufficient space for model files (4-8GB per model)

### Required Software

1. **Docker Desktop**
   - Download from: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - No additional configuration required for GPU support

2. **NVIDIA GPU Drivers**
   - Download the latest drivers for your GPU from: [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)
   - Ensure drivers are up to date for Docker GPU access

3. **Build Tools for Visual Studio 2022** (Windows only)
   - Required for compiling Python packages with native extensions
   - Download from: [https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - During installation, select "Desktop development with C++"
   - This is required for installing `llama-cpp-python` and other packages with C++ dependencies

5. **Python 3.11+**
   - Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Ensure Python is added to your system PATH

6. **Poetry** (Python dependency management)

   ```bash
   pip install poetry
   ```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/brAIniac.git
   cd brAIniac
   ```

2. **Download a model**
   - Create a `models` directory in the project root
   - Download a GGUF format model (e.g., from https://huggingface.co)
   - Place the model file in the `models` directory
   - Example: `luna-ai-llama2-uncensored.Q4_K_M.gguf`

3. **Install Python dependencies**

   ```bash
   poetry install
   ```

4. **Build and run the Docker container**

   ```bash
   docker-compose up --build
   ```

## Usage

### Starting the Server

```bash
docker-compose up
```

The server will be available at `http://localhost:8080`

### Testing the Server

Run the test script to verify the server is working:

```bash
poetry run python test_server.py
```

### Making API Requests

The server exposes an OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

completion = client.chat.completions.create(
    model="local-model",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(completion.choices[0].message.content)
```

## Configuration

### Docker Compose Settings

Edit `docker-compose.yml` to customize:

- Model path: `--model /models/your-model.gguf`
- Context size: `--ctx-size 4096`
- GPU layers: `--n-gpu-layers -1` (-1 for all layers)
- Port: Change `8080:8080` to your preferred port

### Server Parameters

Available llama-server parameters:

- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8080)
- `--ctx-size`: Context window size
- `--n-gpu-layers`: Number of layers to offload to GPU
- `--chat-template`: Chat template format (e.g., chatml)

## Troubleshooting

### CUDA Driver Errors

If you see "CUDA driver is a stub library":

- Ensure NVIDIA drivers are installed on the host
- Verify NVIDIA Container Toolkit is installed in WSL 2
- Check that Docker is using the nvidia runtime

### Build Errors on Windows

If you encounter compilation errors:

- Install Build Tools for Visual Studio 2022
- Ensure "Desktop development with C++" workload is selected
- Restart your terminal after installation

### Out of Memory Errors

If the model runs out of memory:

- Reduce `--ctx-size` in docker-compose.yml
- Use a smaller quantized model (e.g., Q4_K_M instead of Q8_0)
- Reduce `--n-gpu-layers` if needed

## Project Structure

```text
brAIniac/
├── models/                 # Model files (GGUF format)
├── .github/
│   └── copilot-instructions.md
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile             # Docker image definition
├── pyproject.toml         # Poetry dependencies
├── test_server.py         # Server test script
└── README.md             # This file
```

## License

[You can find the license information here](LICENSE)

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Fast LLM inference engine
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) - GPU acceleration
- [Docker](https://www.docker.com/) - Containerization platform
- [Hugging Face](https://huggingface.co/) - Model hosting and sharing platform
- [OpenAI API](https://openai.com/api/) - API specification reference
