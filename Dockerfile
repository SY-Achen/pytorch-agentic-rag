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
# Use a fast China mirror in cloud builds; override with --build-arg if needed.
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com
# Download the CPU wheel with curl: pip's streamed download stalls on this 2C2G host.
ARG PYTORCH_CPU_WHEEL=https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp311-cp311-linux_x86_64.whl
RUN curl -L --retry 5 --retry-delay 3 --connect-timeout 20 -o /tmp/torch-2.5.1+cpu-cp311-cp311-linux_x86_64.whl ${PYTORCH_CPU_WHEEL} \
    && pip install --no-cache-dir /tmp/torch-2.5.1+cpu-cp311-cp311-linux_x86_64.whl \
    && rm -f /tmp/torch-2.5.1+cpu-cp311-cp311-linux_x86_64.whl
RUN pip install --no-cache-dir --index-url ${PIP_INDEX_URL} --trusted-host ${PIP_TRUSTED_HOST} \
    --default-timeout=120 --retries=5 -r requirements.txt

# 2. Create persistent data dirs (models under /app/data so it survives restarts)
RUN mkdir -p /app/data/uploads /app/data/vector_db /app/data/models

# 3. Copy source
COPY server.py ./
COPY index.html ./
COPY seed_ingest.py ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/system/status || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
