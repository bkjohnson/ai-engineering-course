#!/usr/bin/env bash
# Single Render web service: Streamlit is the public process on $PORT,
# the FastAPI backend runs internally on localhost.
# Render start command: bash start-render.sh
set -euo pipefail

INTERNAL_API_PORT="${INTERNAL_API_PORT:-8000}"

uvicorn main:app --host 127.0.0.1 --port "$INTERNAL_API_PORT" &

API_BASE_URL="http://127.0.0.1:$INTERNAL_API_PORT" streamlit run streamlit_app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true &

# Exit when either process exits so the platform restarts the service.
# `wait -n` needs bash >= 4.3 (Render has it); plain `wait` fallback for
# older shells like macOS system bash.
if [ "${BASH_VERSINFO[0]}" -ge 5 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 3 ]; }; then
  wait -n
else
  wait
fi
