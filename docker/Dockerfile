# Stage 1: Base Python image with Poetry
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

# Stage 2: Install dependencies (cached layer)
FROM base AS dependencies

# Copy only dependency files to leverage Docker layer caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies without the project package itself
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Stage 3: Final application image
FROM dependencies AS final

# Copy application source code
COPY core/ ./core/
COPY servers/ ./servers/
COPY main.py ./

# Install the project package
RUN poetry install --only main --no-interaction --no-ansi

# Entry point - run the CLI
CMD ["python", "main.py"]
