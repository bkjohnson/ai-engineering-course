#!/usr/bin/env bash
# Single Render web service: the gateway (uvicorn) is the public process on
# $PORT. It serves the API natively and reverse-proxies everything else to
# the internal Streamlit UI, so both are publicly reachable on one URL.
# Render start command: bash start-render.sh
set -euo pipefail

UI_INTERNAL_PORT="${UI_INTERNAL_PORT:-8501}"
PUBLIC_PORT="${PORT:-8000}"

# Internal UI: localhost only; the gateway is its sole caller.
API_BASE_URL="http://127.0.0.1:$PUBLIC_PORT" streamlit run streamlit_app.py \
  --server.port "$UI_INTERNAL_PORT" \
  --server.address 127.0.0.1 \
  --server.headless true &

# Public gateway: API + proxy to the UI.
UI_INTERNAL_PORT="$UI_INTERNAL_PORT" uvicorn gateway:app \
  --host 0.0.0.0 --port "$PUBLIC_PORT" &

# Exit when either process exits so the platform restarts the service.
# `wait -n` needs bash >= 4.3 (Render has it); plain `wait` fallback for
# older shells like macOS system bash.
if [ "${BASH_VERSINFO[0]}" -ge 5 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 3 ]; }; then
  wait -n
else
  wait
fi
