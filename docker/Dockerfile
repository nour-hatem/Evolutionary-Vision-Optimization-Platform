FROM python:3.10-slim

WORKDIR /app

# Install curl for healthcheck probe
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY docker/requirements.txt docker/requirements.txt
RUN pip install --no-cache-dir --timeout=200 -r docker/requirements.txt

# Copy application source
COPY api/         /app/api/
COPY model/       /app/model/
COPY data/        /app/data/
COPY experiments/ /app/experiments/

# Copy root-level config/metrics files used by predict.py
COPY best_config.json /app/best_config.json
COPY metrics.json     /app/metrics.json

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
