"""Public gateway: one process that serves the API natively and reverse-proxies
everything else — HTTP and websockets — to the internal Streamlit UI.

Intended for a single public port (e.g. one Render web service):

    streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501 &
    uvicorn gateway:app --host 0.0.0.0 --port $PORT

API routes (/ask, /chat, /docs) are matched first by FastAPI; any other path
falls through to the catch-all routes below and is forwarded to Streamlit,
including the /_stcore/stream websocket the UI depends on.
"""

import asyncio
import logging
import os

import httpx
import websockets
from fastapi import Request, WebSocket
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, StreamingResponse

from main import app

logger = logging.getLogger("gateway")

UI_HOST = os.getenv("UI_INTERNAL_HOST", "127.0.0.1")
UI_PORT = int(os.getenv("UI_INTERNAL_PORT", "8501"))

ui_client = httpx.AsyncClient(
    base_url=f"http://{UI_HOST}:{UI_PORT}",
    # No read timeout: Streamlit holds some HTTP connections open.
    timeout=httpx.Timeout(15.0, read=None),
)

# Headers that must not be blindly forwarded between hops.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host",
}


def _forwardable(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


@app.websocket("/{path:path}")
async def proxy_websocket(client_ws: WebSocket, path: str) -> None:
    # Streamlit streams app state over the /_stcore/stream websocket and
    # carries its session token in the subprotocol list — pass it through.
    requested = client_ws.headers.get("sec-websocket-protocol")
    subprotocols = [p.strip() for p in requested.split(",")] if requested else None

    try:
        upstream = await websockets.connect(
            f"ws://{UI_HOST}:{UI_PORT}/{path}",
            subprotocols=subprotocols,
            max_size=None,
        )
    except Exception:
        logger.exception("Could not open upstream websocket for /%s", path)
        await client_ws.close(code=1014)  # bad gateway
        return

    await client_ws.accept(subprotocol=upstream.subprotocol)

    async def client_to_upstream() -> None:
        while True:
            message = await client_ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("text") is not None:
                await upstream.send(message["text"])
            elif message.get("bytes") is not None:
                await upstream.send(message["bytes"])

    async def upstream_to_client() -> None:
        async for message in upstream:
            if isinstance(message, str):
                await client_ws.send_text(message)
            else:
                await client_ws.send_bytes(message)

    tasks = [
        asyncio.create_task(client_to_upstream()),
        asyncio.create_task(upstream_to_client()),
    ]
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
    finally:
        await upstream.close()


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def proxy_http(request: Request, path: str):
    upstream_request = ui_client.build_request(
        request.method,
        httpx.URL(path=f"/{path}", query=request.url.query.encode()),
        headers=_forwardable(request.headers),
        content=request.stream(),
    )
    try:
        upstream_response = await ui_client.send(upstream_request, stream=True)
    except httpx.TransportError:
        logger.exception("UI upstream unreachable for %s /%s", request.method, path)
        return JSONResponse({"detail": "The UI is unavailable"}, status_code=502)
    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=_forwardable(upstream_response.headers),
        background=BackgroundTask(upstream_response.aclose),
    )
