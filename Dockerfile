# Smart RAG Agent — Production Dockerfile (ponytail)
# Usage:
#   1) Build:    docker build -t smart-rag-agent .
#   2) Run:      docker compose up -d
#   3) Deploy:   scp -r user@server:/opt/rag_agent && cd /opt/rag_agent && docker compose up -d --build

FROM python:3.11-slim

WORKDIR /app

# 1. Install deps + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 2>&1 | tail -5

# 2. Create persistent data dirs (models under /app/data so it survives restarts)
RUN mkdir -p /app/data/uploads /app/data/vector_db /app/data/models

# 3. Copy source
COPY server.py ./
COPY index.html ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/system/status || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
