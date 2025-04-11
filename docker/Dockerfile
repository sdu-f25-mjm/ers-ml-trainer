# Base stage with common dependencies
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC

# Install system dependencies (grouped to reduce layers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Development/build stage
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what's needed for installing dependencies
COPY /requirements.txt /setup.py ./
COPY /ers_ml_trainer/__init__.py ./ers_ml_trainer/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# CPU stage - production ready
FROM base AS cpu

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/cache_eval_results /app/mock /app/logs && \
    chmod -R 775 /app/models /app/cache_eval_results /app/mock /app/logs

# Health check that doesn't require additional dependencies
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]

# GPU stage - based on CPU stage
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS gpu

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install Python and minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    libmariadb-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app

# Copy from builder stage for dependencies
COPY --from=cpu /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=cpu /usr/local/bin /usr/local/bin
COPY --from=builder /build /app


# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models cache_eval_results mock logs && \
    chmod -R 777 models cache_eval_results mock logs

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]