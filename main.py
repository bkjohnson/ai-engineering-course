import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()  # must run before the client reads ANTHROPIC_API_KEY

app = FastAPI(title="Ask Claude")
client = anthropic.Anthropic()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=16000,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": request.question}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="Anthropic API key is invalid")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited by the Anthropic API")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Could not reach the Anthropic API")

    answer = "".join(block.text for block in response.content if block.type == "text")
    return AskResponse(answer=answer)
