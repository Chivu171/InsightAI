# InsightAI Backend — Production Dockerfile (Fly.io)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_CACHE=/data/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/data/hf_cache

WORKDIR /app

# System deps (libgomp for FAISS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer cache: dependencies first ──────────────────────────────
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip uninstall -y langchain langchain-community langchain-core langchain-experimental langchain-huggingface langchain-text-splitters 2>/dev/null || true && \
    pip install --no-cache-dir \
        langchain==0.3.7 \
        langchain-community==0.3.5 \
        langchain-core==0.3.15 \
        langchain-experimental==0.3.3 \
        langchain-huggingface==0.1.1 \
        langchain-text-splitters==0.3.2 && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding models (cached in HF_HOME) ────────────
# On Fly, /data is a persistent volume — first boot downloads, later boots reuse.
# We use a small warmup that tolerates offline (volume already has cache).
RUN mkdir -p /data/hf_cache && \
    python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
try: SentenceTransformer('BAAI/bge-small-en-v1.5'); print('Cached BAAI/bge-small-en-v1.5'); \
except Exception as e: print('HF download skipped (offline/build):', e); \
try: CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('Cached cross-encoder'); \
except Exception as e: print('CrossEncoder download skipped:', e)" || true

# ── App code ─────────────────────────────────────────────────────
COPY . .

# Ensure data dirs exist (overridden by Fly volume mount at runtime)
RUN mkdir -p /data/vector_db /data/uploads /data/hf_cache

EXPOSE 8000

# Healthcheck for local docker (Fly uses http_checks instead).
# /ready gates on embedding model load, so start-period must cover the
# HuggingFace model download + load time on first boot (~60-180s).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')" || exit 1

# Build marker — written at image build time, useful in `fly logs` to
# confirm a fresh deploy actually replaced the running image.
ARG BUILD_SHA=local
RUN echo "build sha=${BUILD_SHA} built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /app/BUILD_INFO && \
    cat /app/BUILD_INFO

# Entrypoint recreates volume-mounted subdirs at every boot (build-time mkdir
# is masked by the volume mount). See entrypoint.sh for the full rationale.
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
