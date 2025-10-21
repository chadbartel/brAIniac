# Start from an NVIDIA CUDA base image that includes the necessary toolkit and Python
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1

# Install build dependencies and git
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    python3-pip \
    curl \
    libcurl4-openssl-dev \
    gpg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the llama.cpp repository
RUN git clone https://github.com/ggerganov/llama.cpp.git

# Install the NVIDIA Container Toolkit for CUDA support
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA support
WORKDIR /app/llama.cpp

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN cmake --build build --config Release

# Set the final working directory to where the server executable is
WORKDIR /app/llama.cpp/build/bin

# Expose the port the server will run on
EXPOSE 8080