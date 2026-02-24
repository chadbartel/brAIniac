@echo off
REM Quick setup script for brAIniac Phase 1 (Windows)

echo ============================================
echo brAIniac Phase 1 - Quick Setup
echo ============================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    exit /b 1
)

echo ✅ Docker found
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy .env.example .env
    echo ✅ .env file created. Edit it to customize your configuration.
) else (
    echo ✅ .env file already exists
)
echo.

REM Start Docker services
echo 🚀 Starting Docker services...
docker compose -f docker/docker-compose.yml up -d

echo.
echo ⏳ Waiting for Ollama to start...
timeout /t 10 /nobreak >nul

echo.
echo 📥 Pulling Llama 3.1 8B model (this may take a few minutes)...
docker exec brainiac-ollama ollama pull llama3.1:8b-instruct-q4_K_M

echo.
echo ============================================
echo ✅ Setup complete!
echo ============================================
echo.
echo To start chatting:
echo   docker attach brainiac-app
echo.
echo Or run locally with Poetry:
echo   poetry install
echo   poetry run python main.py
echo.
echo Helpful commands:
echo   docker compose -f docker/docker-compose.yml logs -f    # View logs
echo   docker compose -f docker/docker-compose.yml down       # Stop services
echo   docker exec brainiac-ollama ollama list                # List models
echo.
pause
