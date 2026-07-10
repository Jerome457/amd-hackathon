import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from router.infer_router import classify
from prompt_profiles import get_prompt_profile

load_dotenv()

from local_llm import generate_with_confidence

LOCAL_CONFIDENCE_THRESHOLD = float(os.getenv("LOCAL_MODEL_FALLBACK_THRESHOLD", "0.65"))

def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=30,
    )


def get_response_with_source(message: str) -> tuple[str, str]:
    # -------------------------
    # Route the prompt
    # -------------------------
    route = classify(message)

    category = route["category"]
    difficulty = route["difficulty"]

    # -------------------------
    # Select model + prompts
    # -------------------------
    profile = get_prompt_profile(category, difficulty)

    if difficulty == "easy":
        try:
            local_result = generate_with_confidence(profile.messages_for(message))
            local_confidence = float(local_result.get("confidence", 0.0))
            should_fallback = bool(local_result.get("fallback_recommended", False))

            if (
                local_result.get("answer")
                and local_confidence >= LOCAL_CONFIDENCE_THRESHOLD
                and not should_fallback
            ):
                return str(local_result["answer"]).strip(), "local_llm"
        except Exception:
            pass

    client = get_client()

    response = client.chat.completions.create(
        model=profile.model,
        temperature=0.0,
        max_tokens=10000,
        messages=profile.messages_for(message),
    )

    return (response.choices[0].message.content or "").strip(), "fireworks"


def get_response(message: str) -> str:
    answer, _ = get_response_with_source(message)
    return answer


def get_answers(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    results = []

    for task in tasks:
        answer = get_response(task["prompt"])

        results.append(
            {
                "task_id": task["task_id"],
                "answer": answer,
            }
        )

    return results