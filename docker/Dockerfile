# Base stage with common dependencies
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb-dev curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Builder stage for dependencies
FROM base AS builder
WORKDIR /build
COPY requirements.txt ./
# Make sure to install all requirements including uvicorn
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# CPU stage - production ready
FROM base AS cpu
WORKDIR /app
# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
RUN pip install --no-cache-dir torch tensorflow-cpu>=2.14.0
COPY . .
CMD ["python", "main.py"]

# GPU stage - production ready
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS gpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip libmariadb-dev curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app
# Copy installed Python packages from builder and fix pip
COPY --from=builder /build/requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch tensorflow>=2.14.0

COPY . .
CMD ["python", "main.py"]