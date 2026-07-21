# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps kept minimal; psycopg[binary] ships its own libpq.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Copy the rest of the project.
COPY configs ./configs
COPY prompts ./prompts
COPY scripts ./scripts
COPY data ./data

# Run as a non-root user.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default: serve the API. Override for CLI/experiments.
CMD ["uvicorn", "jobrec.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
