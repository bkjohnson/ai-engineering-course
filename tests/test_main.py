"""Tests for the deterministic behavior of the service.

The Anthropic client is mocked throughout — no API key or network access is
needed. Answer *quality* is out of scope here (that's for evals).
"""

from contextlib import contextmanager
from unittest.mock import patch

import anthropic
import httpx2 as httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import main

client = TestClient(main.app)

API_URL = "https://api.anthropic.com/v1/messages"


def api_error(cls, status: int, message: str) -> anthropic.APIStatusError:
    request = httpx.Request("POST", API_URL)
    return cls(message, response=httpx.Response(status, request=request), body=None)


def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=httpx.Request("POST", API_URL))


class FakeParsedResponse:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output


GOOD_ANSWER = main.AskResponse(answer="Paris.", sources=["Common knowledge"], confidence=0.98)


# --- /ask ---------------------------------------------------------------


def test_ask_happy_path_uses_primary_model():
    calls = []

    def fake_parse(**kwargs):
        calls.append(kwargs)
        return FakeParsedResponse(GOOD_ANSWER)

    with patch.object(main.client.messages, "parse", side_effect=fake_parse):
        response = client.post("/ask", json={"question": "Capital of France?"})

    assert response.status_code == 200
    body = response.json()
    assert body == {"answer": "Paris.", "sources": ["Common knowledge"], "confidence": 0.98}
    assert len(calls) == 1
    assert calls[0]["model"] == main.PRIMARY_MODEL
    assert calls[0]["thinking"] == {"type": "adaptive"}


def test_ask_falls_back_when_primary_rate_limited():
    calls = []

    def fake_parse(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == main.PRIMARY_MODEL:
            raise api_error(anthropic.RateLimitError, 429, "rate limited")
        return FakeParsedResponse(GOOD_ANSWER)

    with patch.object(main.client.messages, "parse", side_effect=fake_parse):
        response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 200
    assert [c["model"] for c in calls] == [main.PRIMARY_MODEL, main.FALLBACK_MODEL]
    # Haiku 4.5 doesn't support adaptive thinking — must not be sent.
    assert "thinking" not in calls[1]


def test_ask_returns_429_when_both_models_rate_limited():
    with patch.object(
        main.client.messages,
        "parse",
        side_effect=lambda **k: (_ for _ in ()).throw(
            api_error(anthropic.RateLimitError, 429, "rate limited")
        ),
    ):
        response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 429


def test_ask_returns_504_on_timeout():
    with patch.object(main.client.messages, "parse", side_effect=timeout_error()):
        response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 504


def test_ask_returns_500_on_bad_api_key():
    with patch.object(
        main.client.messages,
        "parse",
        side_effect=api_error(anthropic.AuthenticationError, 401, "bad key"),
    ):
        response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 500


def test_ask_returns_502_when_output_unparseable():
    with patch.object(
        main.client.messages, "parse", return_value=FakeParsedResponse(None)
    ):
        response = client.post("/ask", json={"question": "hi"})

    assert response.status_code == 502


def test_ask_honors_requested_model_without_thinking_on_haiku():
    calls = []

    def fake_parse(**kwargs):
        calls.append(kwargs)
        return FakeParsedResponse(GOOD_ANSWER)

    with patch.object(main.client.messages, "parse", side_effect=fake_parse):
        response = client.post(
            "/ask", json={"question": "hi", "model": "claude-haiku-4-5"}
        )

    assert response.status_code == 200
    assert calls[0]["model"] == "claude-haiku-4-5"
    assert "thinking" not in calls[0]


def test_ask_sends_thinking_for_sonnet():
    calls = []

    def fake_parse(**kwargs):
        calls.append(kwargs)
        return FakeParsedResponse(GOOD_ANSWER)

    with patch.object(main.client.messages, "parse", side_effect=fake_parse):
        response = client.post(
            "/ask", json={"question": "hi", "model": "claude-sonnet-5"}
        )

    assert response.status_code == 200
    assert calls[0]["model"] == "claude-sonnet-5"
    assert calls[0]["thinking"] == {"type": "adaptive"}


def test_ask_rejects_unsupported_model():
    response = client.post(
        "/ask", json={"question": "hi", "model": "gpt-4o-mini"}
    )
    assert response.status_code == 422


def test_ask_no_fallback_retry_when_fallback_model_requested():
    calls = []

    def fake_parse(**kwargs):
        calls.append(kwargs)
        raise api_error(anthropic.RateLimitError, 429, "rate limited")

    with patch.object(main.client.messages, "parse", side_effect=fake_parse):
        response = client.post(
            "/ask", json={"question": "hi", "model": main.FALLBACK_MODEL}
        )

    assert response.status_code == 429
    assert len(calls) == 1  # no pointless second attempt on the same model


def test_ask_rejects_invalid_request_body():
    response = client.post("/ask", json={"prompt": "wrong field"})
    assert response.status_code == 422


def test_confidence_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        main.AskResponse(answer="x", sources=[], confidence=1.5)


# --- /chat --------------------------------------------------------------


class FakeStream:
    def __init__(self, chunks):
        self.text_stream = iter(chunks)


def test_chat_happy_path_streams_text():
    @contextmanager
    def fake_stream(**kwargs):
        assert kwargs["model"] == main.PRIMARY_MODEL
        yield FakeStream(["Hello ", "there."])

    with patch.object(main.client.messages, "stream", fake_stream):
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.text == "Hello there."


def test_chat_falls_back_when_primary_fails_before_first_token():
    @contextmanager
    def fake_stream(**kwargs):
        if kwargs["model"] == main.PRIMARY_MODEL:
            raise api_error(anthropic.InternalServerError, 529, "overloaded")
        yield FakeStream(["fallback ", "stream"])

    with patch.object(main.client.messages, "stream", fake_stream):
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.text == "fallback stream"


def test_chat_midstream_failure_keeps_partial_and_adds_notice():
    class DyingStream:
        @property
        def text_stream(self):
            def gen():
                yield "partial "
                raise api_error(anthropic.InternalServerError, 500, "server error")

            return gen()

    @contextmanager
    def fake_stream(**kwargs):
        yield DyingStream()

    with patch.object(main.client.messages, "stream", fake_stream):
        response = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.text.startswith("partial ")
    assert "interrupted" in response.text


def test_chat_rejects_invalid_role():
    response = client.post("/chat", json={"messages": [{"role": "system", "content": "hi"}]})
    assert response.status_code == 422


def test_get_chat_redirects_to_ui():
    response = client.get("/chat", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/"


def test_index_serves_chat_ui():
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>Ask Claude</title>" in response.text
