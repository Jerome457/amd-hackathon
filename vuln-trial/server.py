import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vuln Trial Task Receiver")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskPayload(BaseModel):
    task: dict[str, Any]


@app.post("/tasks")
async def receive_task(payload: TaskPayload, request: Request) -> dict[str, str]:
    record = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "client": {
            "host": request.client.host if request.client else None,
            "port": request.client.port if request.client else None,
        },
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "task": payload.task,
    }

    logger.info(json.dumps(record, ensure_ascii=False))

    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}