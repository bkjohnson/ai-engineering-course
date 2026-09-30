"""Tests for the gateway: API routes stay native, unknown paths are proxied."""

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

# Point the proxy at a port nothing listens on BEFORE importing the gateway,
# so proxy tests don't depend on (or hit) a locally running Streamlit.
os.environ["UI_INTERNAL_PORT"] = "59999"

import gateway  # noqa: E402  (registers the proxy routes on main.app)
import main  # noqa: E402

client = TestClient(gateway.app)


class FakeUsage:
    input_tokens = 100
    output_tokens = 50


class FakeParsedResponse:
    parsed_output = main.ModelAnswer(answer="Paris.", sources=[], confidence=0.9)
    usage = FakeUsage()


def test_api_routes_take_precedence_over_the_proxy():
    with patch.object(main.client.messages, "parse", return_value=FakeParsedResponse()):
        response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Paris."


def test_openapi_docs_stay_native():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_unknown_paths_return_502_when_ui_is_down():
    response = client.get("/")
    assert response.status_code == 502
    assert response.json() == {"detail": "The UI is unavailable"}
