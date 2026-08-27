#!/bin/sh
# InsightAI entrypoint — ensure volume-mounted dirs exist before uvicorn starts.
# On Fly.io, /data is a persistent volume mounted at runtime. Directories created
# at Docker build time (RUN mkdir -p /data/...) are masked by the volume mount.
# This entrypoint recreates them on every boot so the app always has writable paths.
set -e

for dir in \
  "${VECTOR_DB_PATH:-/data/vector_db}" \
  "${UPLOAD_DIR:-/data/uploads}" \
  "${HF_HOME:-/data/hf_cache}"; do
  if ! mkdir -p "$dir" 2>/dev/null; then
    echo "[entrypoint] WARN: cannot mkdir $dir (read-only or permission denied)"
  fi
done

# Uvicorn runs as root in the container, but Fly sometimes restricts /data to uid 1000.
# Try to chown to the same uid that wrote the volume originally.
chown -R 1000:1000 /data 2>/dev/null || true

exec "$@"
