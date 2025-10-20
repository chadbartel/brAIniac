# Start from an NVIDIA CUDA base image that includes the necessary toolkit and Python
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install build dependencies and git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the llama.cpp repository
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Build llama.cpp with CUDA support
WORKDIR /app/llama.cpp
RUN cmake -B build -DGGML_CUDA=ON
RUN cmake --build build --config Release

# Set the final working directory to where the server executable is
WORKDIR /app/llama.cpp/build/bin

# Expose the port the server will run on
EXPOSE 8080