# Start from an official NVIDIA PyTorch image with Python 3.12 and CUDA support
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# Expose the port for the local LLM server (e.g., 8000 for llama.cpp server)
EXPOSE 8000

# Command to run when the container starts (example for running the main application script)
CMD ["python", "main.py"]