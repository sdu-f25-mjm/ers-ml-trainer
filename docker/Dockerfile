FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=UTC \
    DOCKER_ENV=true \
    PYTHONPATH=/app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libmariadb-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file (ensure .dockerignore excludes unnecessary files)
COPY docker/requirements_gpu.txt .
# Create virtual environment and install pre‑built binaries via pip

RUN pip install --no-cache-dir --upgrade pip setuptools wheel pip && \
    pip3 install torch --index-url https://download.pytorch.org/whl/cu126 && \
    pip install --no-cache-dir -r requirements_gpu.txt && \
    rm -rf /root/.cache \
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]