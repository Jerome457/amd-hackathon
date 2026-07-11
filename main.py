import os
from typing import Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from dotenv import load_dotenv
from openai import OpenAI

from router.infer_router import classify
from prompt_profiles import get_prompt_profile
from local_llm import generate_with_confidence

load_dotenv()

LOCAL_TIMEOUT = 10  # seconds
LOCAL_CONFIDENCE_THRESHOLD = float(
    os.getenv("LOCAL_MODEL_FALLBACK_THRESHOLD", "0.5")
)

LOCAL_CATEGORIES = {
    "factual_qa",
    "summarization",
    "sentiment",
    "math_reasoning",
    "debugging",
    "code_generation",
    "ner",
}

executor = ThreadPoolExecutor(max_workers=1)


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=30,
    )


def get_response_with_source(message: str) -> tuple[str, str]:
    route = classify(message)

    category = route["category"]
    difficulty = route["difficulty"]

    profile = get_prompt_profile(category, difficulty)

    if difficulty == "easy" and category in LOCAL_CATEGORIES:
        try:
            future = executor.submit(
                generate_with_confidence,
                profile.messages_for(message),
            )

            try:
                local_result = future.result(timeout=LOCAL_TIMEOUT)
            except TimeoutError:
                future.cancel()
                local_result = None

            if local_result is not None:
                answer = str(local_result.get("answer", "")).strip()

                try:
                    confidence = float(local_result.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0

                fallback = bool(
                    local_result.get("fallback_recommended", False)
                )

                if (
                    answer
                    and answer.lower() != "none"
                    and confidence >= LOCAL_CONFIDENCE_THRESHOLD
                    and not fallback
                ):
                    return answer, "local_llm"

        except Exception:
            pass

    client = get_client()

    response = client.chat.completions.create(
        model=profile.model,
        temperature=0.0,
        max_tokens=10000,
        messages=profile.messages_for(message),
    )

    return (response.choices[0].message.content or ""), "fireworks"


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