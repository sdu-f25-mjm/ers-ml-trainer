FROM python:3.11-slim-bullseye
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    PYTHONPATH=/app \
    DOCKER_ENV=true

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libmariadb-dev && \
    rm -rf /var/lib/apt/lists/*

COPY docker/requirements_cpu.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_cpu.txt && \
    rm -rf /root/.cache \

COPY . .

RUN mkdir -p /app/logs /app/models /app/model_checkpoints /app/cache_eval_results && \
    useradd -m -r -u 1000 appuser && \
    chown -R appuser:appuser /app \

USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]