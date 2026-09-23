from typing import Literal

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()  # must run before the client reads ANTHROPIC_API_KEY

app = FastAPI(title="Ask Claude")
client = anthropic.Anthropic()

SYSTEM_PROMPT = (
    "You are a helpful assistant with a dry wit. Answer questions accurately and "
    "completely, but deliver your answers with understated, deadpan humor where it "
    "fits naturally. Never let the wit get in the way of a clear, correct answer."
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
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


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        response = client.messages.parse(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": request.question}],
            output_format=AskResponse,
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="Anthropic API key is invalid")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the Anthropic API")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the Anthropic API")

    if response.parsed_output is None:
        raise HTTPException(status_code=502, detail="Model returned an unparseable response")
    return response.parsed_output


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    def generate():
        try:
            with client.messages.stream(
                model="claude-opus-4-8",
                max_tokens=64000,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[m.model_dump() for m in request.messages],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.APIError as e:
            # Headers are already sent, so surface the error in the stream body.
            yield f"\n\n[Error from the Anthropic API: {getattr(e, 'message', str(e))}]"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/chat", include_in_schema=False)
def chat_page() -> RedirectResponse:
    # Navigating to /chat in a browser is a GET; the API endpoint is POST-only.
    return RedirectResponse(url="/")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("static/index.html")
