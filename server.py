from fastapi import FastAPI
from pydantic import BaseModel, Field

from main import get_response

app = FastAPI(title="AMD Hackathon Agent")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str


@app.post("/response", response_model=ChatResponse)
def create_response(request: ChatRequest) -> ChatResponse:
    return ChatResponse(response=get_response(request.message))
