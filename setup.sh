#!/bin/bash
# Quick setup script for brAIniac Phase 1

set -e

echo "============================================"
echo "brAIniac Phase 1 - Quick Setup"
echo "============================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created. Edit it to customize your configuration."
else
    echo "✅ .env file already exists"
fi
echo ""

# Start Docker services
echo "🚀 Starting Docker services..."
docker compose -f docker/docker-compose.yml up -d

echo ""
echo "⏳ Waiting for Ollama to start..."
sleep 10

echo ""
echo "📥 Pulling Llama 3.1 8B model (this may take a few minutes)..."
docker exec brainiac-ollama ollama pull llama3.1:8b-instruct-q4_K_M

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo "============================================"
echo ""
echo "To start chatting:"
echo "  docker attach brainiac-app"
echo ""
echo "Or run locally with Poetry:"
echo "  poetry install"
echo "  poetry run python main.py"
echo ""
echo "Helpful commands:"
echo "  docker compose -f docker/docker-compose.yml logs -f    # View logs"
echo "  docker compose -f docker/docker-compose.yml down       # Stop services"
echo "  docker exec brainiac-ollama ollama list                # List models"
echo ""
