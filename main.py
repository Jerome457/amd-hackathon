import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from prompt_profiles import get_default_model, get_prompt_profile
from router import route_message

load_dotenv()


def get_model() -> str:
    return get_default_model()


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=30,
    )


def get_response(message: str, model: str | None = None) -> str:
    client = get_client()
    route = route_message(client, message, model or get_model())
    profile = get_prompt_profile(route.category, route.difficulty)

    response = client.chat.completions.create(
        model=model or profile.model,
        max_tokens=10000,
        temperature=0.0,
        messages=profile.messages_for(message),
    )

    return (response.choices[0].message.content or "").strip()


def get_answers(tasks: list[dict[str, Any]], model: str | None = None) -> list[dict[str, str]]:
    results = []

    for task in tasks:
        results.append(
            {
                "task_id": task["task_id"],
                "answer": get_response(task["prompt"], model=model),
            }
        )

    return results
