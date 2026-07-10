import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

LOG_PATH = Path(os.getenv("VULN_TRIAL_LOG_PATH", "received_tasks.jsonl"))

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


def append_record(record: dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    append_record(record)
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
