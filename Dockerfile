# SupportDeskEnv - OpenEnv environment for B2B SaaS support ticket resolution
#
# Supports two build modes:
#   1. Official OpenEnv base (recommended for HF Spaces):
#      docker build --build-arg BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest -t support-desk-env .
#   2. Self-contained from Python (fallback, no external base needed):
#      docker build -t support-desk-env .
#
# Hugging Face Spaces uses app_port from README.md front-matter (default: 7860).
# OpenEnv defaults to port 8000; HF Spaces maps its external port internally.

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose OpenEnv default port (HF Spaces will map this via app_port)
EXPOSE 8000

# Respect PORT env var (HF Spaces sets this)
ENV PORT=8000
ENV HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

CMD uvicorn server.app:app --host ${HOST} --port ${PORT} --workers ${WORKERS:-1}
