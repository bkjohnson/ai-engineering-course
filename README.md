# Ask Claude

A FastAPI backend and Streamlit chat UI for asking questions of Anthropic's
Claude models — helpful answers, dry wit included at no extra charge.

## What's here

| File | Purpose |
|---|---|
| `main.py` | FastAPI backend: `/ask` (structured JSON answers) and `/chat` (streaming) |
| `streamlit_app.py` | Streamlit chat UI; talks to the API's `/chat` endpoint |
| `gateway.py` | Reverse proxy for single-service deploys: serves the API natively, forwards everything else (HTTP + websockets) to Streamlit |
| `Procfile` | Runs the API + UI together locally via `honcho start` |
| `start-render.sh` | Start script for a single Render web service (gateway + internal UI) |
| `tests/` | Pytest suite (fully mocked — no API key or network needed) |

### API endpoints

- `POST /ask` — `{"question": "...", "model": "claude-haiku-4-5"}` (model optional).
  Returns `{answer, sources, confidence, tokens_used, cost_usd}`.
- `POST /chat` — `{"messages": [{"role": "user", "content": "..."}]}`.
  Streams the reply as plain text; used by the Streamlit UI.
- Interactive docs at `/docs`.

Both endpoints retry transient failures and fall back to `claude-haiku-4-5`
if the requested model is unavailable.

## Running locally

Requirements: Python 3.11+ and an Anthropic API key.

```bash
# 1. One-time setup
python3 -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install -r requirements.txt

# 2. Create a .env file in the project root containing your key:
#    ANTHROPIC_API_KEY=<your key>
#    (.env is gitignored — never commit it)

# 3. Start both the API and the UI
honcho start
```

- Chat UI: http://localhost:8501
- API: http://127.0.0.1:8000 (docs at `/docs`)

Ctrl+C stops both. Ports are overridable: `API_PORT=9000 UI_PORT=9501 honcho start`
(the UI automatically points at whatever port the API uses).

You can also run either process on its own:

```bash
uvicorn main:app --reload                 # API only
streamlit run streamlit_app.py            # UI only (expects the API on :8000,
                                          # override with API_BASE_URL)
```

### Tests

```bash
python -m pytest
```

The suite mocks the Anthropic client entirely, so it runs with no API key,
no network access, and no cost.

## Deploying to Render

The app runs as a **single web service** — the UI and the API are both
publicly reachable on the same URL via a Python reverse proxy (`gateway.py`).

Render service settings:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `bash start-render.sh`
- **Environment variable:** `ANTHROPIC_API_KEY` (set it in the Render
  dashboard — it is never committed to the repo)

How it works: Render routes all public traffic to a single `$PORT`, so
`start-render.sh` puts the gateway (uvicorn) there. The gateway serves the
API routes (`/ask`, `/chat`, `/docs`) natively and reverse-proxies every
other path — Streamlit's pages, static assets, and its `/_stcore/stream`
websocket — to a Streamlit process bound to localhost inside the same
container. If either process exits, the script exits so Render restarts
the service.

So on the deployed URL:

- `/` → the chat UI
- `POST /ask`, `POST /chat`, `/docs` → the API, exactly as when run alone
