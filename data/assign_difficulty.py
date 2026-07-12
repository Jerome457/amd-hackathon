import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from local_llm import get_llm

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "synthetic_dataset.json"
OUTPUT_FILE = BASE_DIR.parent / "router_training_dataset.json"

MODEL = "accounts/fireworks/models/minimax-m3"

client = OpenAI(
    api_key=os.environ["FIREWORKS_API_KEY"],
    base_url=os.environ["FIREWORKS_BASE_URL"],
)

llm = get_llm()


def generate_local_answer(prompt: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=512,
    )

    return response["choices"][0]["message"]["content"].strip()


JUDGE_PROMPT = """
You are evaluating whether a SMALL LOCAL LANGUAGE MODEL solved a task.

Given:

1. User Prompt
2. Ground Truth
3. Model Answer

Determine whether the answer is MOSTLY CORRECT.

Use these rules:

- Minor wording differences are OK.
- Equivalent answers are OK.
- Different formatting is OK.
- Ignore grammar.
- Ignore extra explanation.

Return:

easy
    if the answer would reasonably be considered correct.

hard
    if it is incorrect, incomplete, hallucinated,
    mathematically wrong,
    misses key information,
    or fails formatting requirements.

Return ONLY one word:

easy

or

hard
"""


def judge(prompt, gt, answer):

    user = f"""
Prompt:
{prompt}

Ground Truth:
{gt}

Model Answer:
{answer}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": JUDGE_PROMPT
            },
            {
                "role": "user",
                "content": user
            }
        ],
        service_tier="priority"
    )

    result = response.choices[0].message.content.lower().strip()

    if "easy" in result:
        return "easy"

    return "hard"


def main():

    # Load input dataset
    with open(INPUT_FILE) as f:
        dataset = json.load(f)

    total = len(dataset)

    # Load existing output if it exists
    existing = []

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = []

    output = existing
    start_idx = len(existing)

    print(f"Resuming from prompt {start_idx}/{total}")

    # Resume from the first unfinished prompt
    for i in range(start_idx, total):

        sample = dataset[i]

        prompt = sample["prompt"]
        gt = sample["ground_truth"]

        try:
            answer = generate_local_answer(prompt)

            difficulty = judge(
                prompt,
                gt,
                answer,
            )

        except Exception as e:
            print(e)
            answer = ""
            difficulty = "hard"

        output.append(
            {
                "prompt": prompt,
                "ground_truth": gt,
                "local_answer": answer,
                "category": sample["category"],
                "difficulty": difficulty,
            }
        )

        print(
            f"[{i + 1}/{total}] {sample['category']:18s} -> {difficulty}"
        )

        # Save after every prompt so you can resume again if interrupted
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)

    print("\nDone.")
    print("Saved to", OUTPUT_FILE)

if __name__ == "__main__":
    main()