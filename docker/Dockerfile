# Base stage with CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    DOCKER_ENV=true \
    PYTHONPATH=/app
WORKDIR /app

# Builder stage: install build dependencies, create virtual environment, install Python packages
FROM base AS builder
WORKDIR /build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        build-essential \
        curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Copy only the requirements file (ensure .dockerignore excludes unnecessary files)
COPY requirements_gpu.txt .
# Create virtual environment and install pre‑built binaries via pip
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements_gpu.txt && \
    rm -rf /root/.cache

# Final stage: production image with runtime dependencies
FROM base AS final
# Install minimal runtime packages and create python symlinks in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip
# Copy optimized virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Copy only application code (use a proper .dockerignore to exclude extra files)
COPY . .
# Create necessary directories and a non‑root user in one RUN command
RUN mkdir -p /app/logs /app/models /app/model_checkpoints /app/cache_eval_results && \
    useradd -m -r -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]