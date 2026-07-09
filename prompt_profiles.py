import os
from dataclasses import dataclass

Category = str
Difficulty = str

GENERAL_MODEL_KEYWORDS = ("minimax", "gemma")
REASONING_CODE_MODEL_KEYWORDS = ("kimi", "code")


@dataclass(frozen=True)
class PromptProfile:
    category: Category
    difficulty: Difficulty
    system_prompt: str
    output_style: str
    model: str = ""
    model_role: str = "general"

    def messages_for(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\nOutput style: {self.output_style}",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]


PROMPT_PROFILES: dict[Category, dict[Difficulty, PromptProfile]] = {
    "factual_qa": {
        "easy": PromptProfile(
            category="factual_qa",
            difficulty="easy",
            model_role="general",
            system_prompt=(
                "Answer factual questions directly. Use only well-established facts. "
                "If the question has multiple parts, answer each part in order."
            ),
            output_style="Return one concise sentence. Do not add extra context.",
        ),
        "hard": PromptProfile(
            category="factual_qa",
            difficulty="hard",
            model_role="general",
            system_prompt=(
                "Answer factual questions carefully and verify relationships between "
                "entities before responding. Avoid speculation."
            ),
            output_style="Return a compact answer with only the facts needed.",
        ),
    },
    "math_reasoning": {
        "easy": PromptProfile(
            category="math_reasoning",
            difficulty="easy",
            model_role="reasoning_code",
            system_prompt=(
                "Solve arithmetic and word problems exactly. Track quantities and "
                "operations carefully before giving the final result."
            ),
            output_style="Return the final number with a brief calculation.",
        ),
        "hard": PromptProfile(
            category="math_reasoning",
            difficulty="hard",
            model_role="reasoning_code",
            system_prompt=(
                "Solve multi-step reasoning problems exactly. Identify assumptions, "
                "perform calculations step by step internally, and check the final value."
            ),
            output_style="Return the final answer plus the shortest useful derivation.",
        ),
    },
    "sentiment": {
        "easy": PromptProfile(
            category="sentiment",
            difficulty="easy",
            model_role="general",
            system_prompt=(
                "Classify the sentiment of the provided text. Consider mixed evidence "
                "when positive and negative statements both appear."
            ),
            output_style="Return one label: positive, negative, neutral, or mixed.",
        ),
        "hard": PromptProfile(
            category="sentiment",
            difficulty="hard",
            model_role="general",
            system_prompt=(
                "Classify nuanced sentiment. Separate factual statements from opinions "
                "and account for contrastive wording."
            ),
            output_style="Return the label and a short reason.",
        ),
    },
    "summarization": {
        "easy": PromptProfile(
            category="summarization",
            difficulty="easy",
            model_role="general",
            system_prompt=(
                "Summarize the source text faithfully. Preserve the central cause, "
                "constraint, or conclusion without adding outside information."
            ),
            output_style="Return exactly one sentence.",
        ),
        "hard": PromptProfile(
            category="summarization",
            difficulty="hard",
            model_role="general",
            system_prompt=(
                "Compress dense technical text while preserving the main bottleneck, "
                "response, and constraints. Do not introduce unsupported claims."
            ),
            output_style="Return exactly one concise sentence.",
        ),
    },
    "ner": {
        "easy": PromptProfile(
            category="ner",
            difficulty="easy",
            model_role="general",
            system_prompt=(
                "Extract named entities from the text and assign clear entity types. "
                "Do not include generic nouns or unnamed references."
            ),
            output_style='Return JSON only, as [{"text": "...", "type": "..."}].',
        ),
        "hard": PromptProfile(
            category="ner",
            difficulty="hard",
            model_role="general",
            system_prompt=(
                "Extract named entities with precise types. Include people, "
                "organizations, locations, dates, products, and events when present."
            ),
            output_style='Return JSON only, as [{"text": "...", "type": "..."}].',
        ),
    },
    "debugging": {
        "easy": PromptProfile(
            category="debugging",
            difficulty="easy",
            model_role="reasoning_code",
            system_prompt=(
                "Find the bug in the provided code and provide the minimal corrected "
                "version. Prefer simple, idiomatic Python."
            ),
            output_style="Return the fixed code first, then one short explanation.",
        ),
        "hard": PromptProfile(
            category="debugging",
            difficulty="hard",
            model_role="reasoning_code",
            system_prompt=(
                "Diagnose code defects precisely. Consider edge cases, empty inputs, "
                "type assumptions, and expected behavior before proposing a fix."
            ),
            output_style="Return corrected code and a concise explanation of the bug.",
        ),
    },
    "logical_reasoning": {
        "easy": PromptProfile(
            category="logical_reasoning",
            difficulty="easy",
            model_role="reasoning_code",
            system_prompt=(
                "Solve logic puzzles by applying each constraint exactly. Eliminate "
                "impossible assignments before answering."
            ),
            output_style="Return the answer and one sentence explaining why.",
        ),
        "hard": PromptProfile(
            category="logical_reasoning",
            difficulty="hard",
            model_role="reasoning_code",
            system_prompt=(
                "Solve constraint reasoning tasks carefully. Track all entities, "
                "attributes, and exclusions before producing the final answer."
            ),
            output_style="Return the final answer with a compact reasoning trace.",
        ),
    },
    "code_generation": {
        "easy": PromptProfile(
            category="code_generation",
            difficulty="easy",
            model_role="reasoning_code",
            system_prompt=(
                "Write correct, simple Python code for the requested behavior. Include "
                "edge-case handling when the prompt asks for it."
            ),
            output_style="Return code only unless the prompt explicitly asks for explanation.",
        ),
        "hard": PromptProfile(
            category="code_generation",
            difficulty="hard",
            model_role="reasoning_code",
            system_prompt=(
                "Write robust Python code with clear handling of edge cases, duplicates, "
                "invalid inputs, and expected return values."
            ),
            output_style="Return code first, followed by minimal notes only if useful.",
        ),
    },
}

CATEGORIES = list(PROMPT_PROFILES)
DIFFICULTY = ["easy", "hard"]


def get_allowed_models() -> list[str]:
    models = os.getenv("ALLOWED_MODELS", "")
    return [model.strip() for model in models.split(",") if model.strip()]


def find_model_by_keywords(models: list[str], keywords: tuple[str, ...]) -> str | None:
    for model in models:
        normalized_model = model.lower()
        if any(keyword in normalized_model for keyword in keywords):
            return model
    return None


def require_allowed_models() -> list[str]:
    allowed_models = get_allowed_models()
    if not allowed_models:
        raise ValueError("ALLOWED_MODELS must contain at least one model")
    return allowed_models


def get_default_model() -> str:
    allowed_models = require_allowed_models()
    return find_model_by_keywords(allowed_models, GENERAL_MODEL_KEYWORDS) or allowed_models[0]


def get_reasoning_code_model() -> str:
    allowed_models = require_allowed_models()
    return find_model_by_keywords(allowed_models, REASONING_CODE_MODEL_KEYWORDS) or allowed_models[0]


def get_code_model() -> str:
    return get_reasoning_code_model()


def recommend_model(category: Category, difficulty: Difficulty = "easy") -> str:
    if category not in PROMPT_PROFILES:
        raise ValueError(f"Unknown category: {category}")
    if difficulty not in PROMPT_PROFILES[category]:
        raise ValueError(f"Unknown difficulty for {category}: {difficulty}")

    if PROMPT_PROFILES[category][difficulty].model_role == "reasoning_code":
        return get_reasoning_code_model()

    return get_default_model()


def get_prompt_profile(
    category: Category,
    difficulty: Difficulty = "easy",
) -> PromptProfile:
    profile = PROMPT_PROFILES[category][difficulty]
    return PromptProfile(
        category=profile.category,
        difficulty=profile.difficulty,
        model=recommend_model(category, difficulty),
        system_prompt=profile.system_prompt,
        output_style=profile.output_style,
        model_role=profile.model_role,
    )


def build_messages(
    user_prompt: str,
    category: Category,
    difficulty: Difficulty = "easy",
) -> list[dict[str, str]]:
    return get_prompt_profile(category, difficulty).messages_for(user_prompt)
