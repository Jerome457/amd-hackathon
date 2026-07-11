import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from llama_cpp import Llama


def _get_model_path() -> Path:
    return Path(os.getenv("LOCAL_MODEL_PATH", "models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"))


def _get_context_size() -> int:
    return int(os.getenv("LOCAL_MODEL_CTX", "4096"))


def _get_threads() -> int:
    return int(os.getenv("LOCAL_MODEL_THREADS", "4"))


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    return Llama(
        model_path=str(_get_model_path()),
        n_ctx=_get_context_size(),
        n_threads=_get_threads(),
        verbose=False,
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_with_confidence(
    messages: list[dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict[str, Any]:
    guarded_messages = list(messages)
    guarded_messages.append(
        {
            "role": "system",
            "content": (
                "Return a JSON object with keys answer, confidence, and "
                "fallback_recommended. answer must be the final response only. "
                "confidence must be a number from 0.0 to 1.0. "
                "Set fallback_recommended to true when you are unsure or the "
                "task exceeds your confidence."
            ),
        }
    )

    output = get_llm().create_chat_completion(
        messages=guarded_messages,
        temperature=temperature,
    )

    content = output["choices"][0]["message"]["content"] or ""
    parsed = _extract_json_object(content)

    if parsed is None:
        return {
            "answer": content.strip(),
            "confidence": 0.0,
            "fallback_recommended": True,
            "raw_content": content,
        }

    answer = "" if parsed.get("answer") is None else str(parsed["answer"]).strip()
    raw_confidence = parsed.get("confidence", 0.0)

    try:
        confidence = float(raw_confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "answer": answer,
        "confidence": max(0.0, min(1.0, confidence)),
        "fallback_recommended": bool(parsed.get("fallback_recommended", False)),
        "raw_content": content,
    }


if __name__ == "__main__":
    sample = generate_with_confidence(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Who invented Python?"},
        ]
    )
    print(sample["answer"])