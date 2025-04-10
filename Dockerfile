FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    build-essential \
    libmariadb-dev \
    git \
    wget \
    nvidia-utils-525 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Fix the symbolic link to use Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip /usr/bin/pip

WORKDIR /app

# Copy the entire application (including setup.py)
COPY . .

# Install the package using setup.py
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install mysql-connector-python==8.0.33 && \
    pip install -e .

# Create necessary directories
RUN mkdir -p models cache_eval_results mock logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application using the entry point defined in setup.py
CMD ["cache-rl-api"]

# Start with a slim Python image
FROM python:3.11-slim as cpu

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install only essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmariadb-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p models cache_eval_results mock logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py"]