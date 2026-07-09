import json
from dataclasses import dataclass
from typing import Any

from prompt_profiles import CATEGORIES, DIFFICULTY, Category, Difficulty


@dataclass(frozen=True)
class Route:
    category: Category
    difficulty: Difficulty


ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier. Classify the user's prompt into exactly one "
    "category and one difficulty. Return JSON only. Do not answer the prompt."
)


def build_router_prompt(message: str) -> str:
    return (
        "Classify this prompt for an AI task router.\n\n"
        f"Allowed categories: {', '.join(CATEGORIES)}\n"
        f"Allowed difficulties: {', '.join(DIFFICULTY)}\n\n"
        "Use difficulty='easy' unless the prompt clearly requires long, multi-step, "
        "ambiguous, or advanced reasoning.\n\n"
        'Return exactly this JSON shape: {"category": "...", "difficulty": "..."}\n\n'
        f"Prompt:\n{message}"
    )


def parse_route(raw_route: str) -> Route:
    raw_route = raw_route.strip()
    if raw_route.startswith("```"):
        raw_route = raw_route.removeprefix("```json").removeprefix("```").strip()
        raw_route = raw_route.removesuffix("```").strip()

    try:
        data = json.loads(raw_route)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Router returned invalid JSON: {raw_route}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Router returned non-object JSON: {raw_route}")

    category = data.get("category")
    difficulty = data.get("difficulty", "easy")
    return validate_route(category, difficulty)


def validate_route(category: Any, difficulty: Any) -> Route:
    if category not in CATEGORIES:
        raise ValueError(f"Router returned unknown category: {category}")
    if difficulty not in DIFFICULTY:
        raise ValueError(f"Router returned unknown difficulty: {difficulty}")

    return Route(category=category, difficulty=difficulty)


def route_message(client: Any, message: str, model: str) -> Route:
    response = client.chat.completions.create(
        model=model,
        max_tokens=100,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_router_prompt(message),
            },
        ],
    )

    raw_route = (response.choices[0].message.content or "").strip()
    return parse_route(raw_route)
