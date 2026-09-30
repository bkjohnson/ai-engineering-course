import logging
from typing import Literal

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()  # must run before the client reads ANTHROPIC_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ask_claude")

PRIMARY_MODEL = "claude-opus-4-8"
FALLBACK_MODEL = "claude-haiku-4-5"
SupportedModel = Literal["claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"]
ADAPTIVE_THINKING_MODELS = {"claude-opus-4-8", "claude-sonnet-5"}  # not Haiku 4.5

# (input, output) USD per million tokens — keep in sync with anthropic.com/pricing
MODEL_PRICING_PER_MTOK = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 2  # SDK retries 429/5xx/connection errors with exponential backoff

# Errors where trying the fallback model is worthwhile (capacity/transient issues).
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)

app = FastAPI(title="Ask Claude")
client = anthropic.Anthropic(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)

SYSTEM_PROMPT = (
    "You are a helpful assistant with a dry wit. Answer questions accurately and "
    "completely, but deliver your answers with understated, deadpan humor where it "
    "fits naturally. Never let the wit get in the way of a clear, correct answer."
)


class AskRequest(BaseModel):
    question: str
    model: SupportedModel = PRIMARY_MODEL


class ModelAnswer(BaseModel):
    """The part of the response the model produces (structured output schema)."""

    answer: str = Field(description="The complete answer to the user's question.")
    sources: list[str] = Field(
        description=(
            "Sources supporting the answer, drawn from your training knowledge "
            "(e.g. named publications, standards, or authoritative references). "
            "Empty if the answer needs no sourcing."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are in the accuracy of the answer, as a number "
            "from 0.0 (pure guess) to 1.0 (certain)."
        ),
    )


class AskResponse(ModelAnswer):
    """The full API response: the model's answer plus server-computed usage."""

    tokens_used: int
    cost_usd: float


def _cost_usd(model: str, usage) -> float:
    input_price, output_price = MODEL_PRICING_PER_MTOK[model]
    cost = (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
    return round(cost, 6)


def _parse_answer(question: str, model: str) -> anthropic.types.Message:
    kwargs = dict(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
        output_format=ModelAnswer,
    )
    if model in ADAPTIVE_THINKING_MODELS:
        kwargs["thinking"] = {"type": "adaptive"}
    return client.messages.parse(**kwargs)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    served_model = request.model
    try:
        try:
            response = _parse_answer(request.question, request.model)
        except RETRYABLE_ERRORS as e:
            if request.model == FALLBACK_MODEL:
                raise  # already on the fallback model; nothing left to try
            logger.warning(
                "Requested model %s unavailable (%s); falling back to %s",
                request.model, type(e).__name__, FALLBACK_MODEL,
            )
            served_model = FALLBACK_MODEL
            response = _parse_answer(request.question, FALLBACK_MODEL)
    except anthropic.AuthenticationError:
        logger.exception("Anthropic authentication failed")
        raise HTTPException(status_code=500, detail="Anthropic API key is invalid")
    except anthropic.RateLimitError:
        logger.warning("Rate limited (requested model: %s)", request.model)
        raise HTTPException(status_code=429, detail="Rate limited by the Anthropic API")
    except anthropic.APITimeoutError:
        logger.exception("Anthropic request timed out after %ss", REQUEST_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="The model took too long to respond")
    except anthropic.APIStatusError as e:
        logger.exception("Anthropic API error")
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e.message}")
    except anthropic.APIConnectionError:
        logger.exception("Could not reach the Anthropic API")
        raise HTTPException(status_code=502, detail="Could not reach the Anthropic API")

    if response.parsed_output is None:
        logger.error("Model returned an unparseable structured response")
        raise HTTPException(status_code=502, detail="Model returned an unparseable response")

    usage = response.usage
    return AskResponse(
        **response.parsed_output.model_dump(),
        tokens_used=usage.input_tokens + usage.output_tokens,
        cost_usd=_cost_usd(served_model, usage),
    )


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    messages = [m.model_dump() for m in request.messages]

    def stream_model(model: str):
        kwargs = dict(model=model, max_tokens=64000, system=SYSTEM_PROMPT, messages=messages)
        if model == PRIMARY_MODEL:
            kwargs["thinking"] = {"type": "adaptive"}  # not supported on Haiku 4.5
        with client.messages.stream(**kwargs) as stream:
            yield from stream.text_stream

    def generate():
        # Headers are already sent once we start yielding, so failures are
        # surfaced in the stream body rather than as HTTP status codes.
        emitted = False
        try:
            for text in stream_model(PRIMARY_MODEL):
                emitted = True
                yield text
            return
        except RETRYABLE_ERRORS as e:
            if emitted:
                logger.exception("Stream failed mid-response on %s", PRIMARY_MODEL)
                yield "\n\n[The response was interrupted by an upstream error — this may be incomplete.]"
                return
            logger.warning(
                "Primary model %s unavailable (%s); falling back to %s",
                PRIMARY_MODEL, type(e).__name__, FALLBACK_MODEL,
            )
        except anthropic.APIError as e:
            logger.exception("Chat request failed on %s", PRIMARY_MODEL)
            yield f"\n\n[Error from the Anthropic API: {getattr(e, 'message', str(e))}]"
            return

        try:
            for text in stream_model(FALLBACK_MODEL):
                yield text
        except anthropic.APIError as e:
            logger.exception("Fallback model %s also failed", FALLBACK_MODEL)
            yield f"\n\n[Error from the Anthropic API: {getattr(e, 'message', str(e))}]"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")
