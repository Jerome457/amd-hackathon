import json
import os
import re
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
# ==========================
# Configuration
# ==========================

load_dotenv()
MODEL = "accounts/fireworks/models/minimax-m3"

PROMPTS_PER_CALL = 10
CALLS_PER_CATEGORY = 5

OUTPUT_FILE = "synthetic_dataset.json"

CATEGORIES = [
    "factual_qa",
    "math_reasoning",
    "sentiment",
    "summarization",
    "ner",
    "debugging",
    "logical_reasoning",
    "code_generation",
]

# ==========================
# Fireworks Client Setup
# ==========================

CATEGORY_HINTS = {
    "code_generation":
        "Generate realistic software engineering tasks including algorithms, APIs, web backends, databases, networking, machine learning, concurrency, file systems, testing, and optimization.",

    "debugging":
        "Generate prompts asking to fix broken code. Include Python, C++, JavaScript, runtime bugs, edge cases, performance issues.",

    "math_reasoning":
        "Generate arithmetic, algebra, geometry, probability, combinatorics and optimization problems.",

    "logical_reasoning":
        "Generate deduction puzzles, scheduling problems, seating arrangements, truth-teller puzzles and constraint reasoning.",

    "factual_qa":
        "Generate questions covering history, geography, biology, chemistry, physics, economics, politics, astronomy, sports, literature, medicine, technology, and current scientific knowledge.",
    "ner":
        "Generate paragraphs containing many people, organizations, locations, products, events and dates.",

    "summarization":
        "Generate prompts requesting summaries of news articles, reports, scientific papers and long passages.",

    "sentiment":
        "Generate reviews, tweets, emails, customer feedback and social media posts."
}

api_key = os.getenv("FIREWORKS_API_KEY")
base_url = os.getenv("FIREWORKS_BASE_URL")

if api_key is None:
    raise RuntimeError("FIREWORKS_API_KEY environment variable not set")

if base_url is None:
    raise RuntimeError("FIREWORKS_BASE_URL environment variable not set")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# ==========================
# System Prompt
# ==========================

SYSTEM_PROMPT = """
You are creating a dataset for training an AI routing classifier.

Return ONLY valid JSON.

Return a JSON array.

Each element MUST have exactly this format:

{
  "prompt": "...",
  "category": "...",
  "ground_truth": "..."
}

Requirements:

- Generate exactly 10 prompt/answer pairs.
- All prompts must belong to the requested category.
- The ground_truth must be completely correct.
- Do NOT explain the answer.
- Do NOT include reasoning unless explicitly requested in the prompt.
- Make prompts realistic as if written by actual users.
- Mix short, medium and long prompts.
- At least 2 prompts should exceed 200 words.
- Avoid duplicate topics.

Category-specific rules:

Factual QA:
- ground_truth should be the factual answer.

Math Reasoning:
- ground_truth should contain the correct final answer.
- Include calculations only if the prompt explicitly asks to show work.

Sentiment:
- ground_truth should contain the correct sentiment label and requested explanation.

Summarization:
- ground_truth should be a valid summary satisfying all constraints.

NER:
- ground_truth should contain every entity with its correct type.

Debugging:
- ground_truth should contain the corrected code.

Logical Reasoning:
- ground_truth should contain the correct conclusion.

Code Generation:
- ground_truth should contain the requested working code.

Return ONLY valid JSON.
"""
# ==========================
# Initialize Data & Resume Tracking
# ==========================

dataset = []
seen = set()

# Optional: Load existing progress if the script crashed earlier
if Path(OUTPUT_FILE).exists():
    try:
        with open(OUTPUT_FILE, "r") as f:
            dataset = json.load(f)
            for item in dataset:
                if isinstance(item, dict) and "prompt" in item:
                    seen.add(item["prompt"].lower().strip())
        print(f"🔄 Resuming pipeline. Loaded {len(dataset)} existing unique prompts from {OUTPUT_FILE}.")
    except Exception as e:
        print(f"⚠️ Could not load existing file, starting fresh. Error: {e}")

# ==========================
# Generate Loop
# ==========================

for category in CATEGORIES:

    print(f"\nGenerating {category}")

    for call in range(CALLS_PER_CATEGORY):

        user_prompt = f"""
        Generate exactly {PROMPTS_PER_CALL} prompt/ground_truth pairs.

        Category:
        {category}

        Category guidance:
        {CATEGORY_HINTS[category]}

        Return ONLY valid JSON.
        """

        text = ""  # Scope initialization for exception logging
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=1.0,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                service_tier="priority"
            )

            text = response.choices[0].message.content.strip()

            # Remove accidental markdown wrap if present
            text = re.sub(r"^```json", "", text, flags=re.IGNORECASE)
            text = re.sub(r"```$", "", text)
            text = text.strip()

            items = json.loads(text)

            if not isinstance(items, list):
                raise ValueError("Model did not return a JSON array/list.")

            # Deduplicate on the fly before adding to dataset
            new_additions = 0
            for item in items:
                if isinstance(item, dict) and "prompt" in item:
                    prompt_normalized = item["prompt"].lower().strip()
                    if prompt_normalized not in seen:
                        seen.add(prompt_normalized)
                        dataset.append(item)
                        new_additions += 1

            print(
                f"{category:20s} "
                f"Batch {call+1:02d}/{CALLS_PER_CATEGORY} "
                f"Added={new_additions} Total={len(dataset)}"
            )

            # Force immediate write and flush to drive
            print(f"Saving to: {Path(OUTPUT_FILE).resolve()}")

            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())  # Forces the kernel to flush buffers to file storage

        except json.JSONDecodeError as je:
            print(f"❌ JSON Parsing Failed on batch {call+1}. The model output wasn't valid JSON.")
            print(f"Raw text preview: {text[:200]}...")
        except Exception as e:
            print(f"❌ API Call Failed on batch {call+1}: {e}")

        time.sleep(1)

print("\nFinished!")
print("Saved a total of", len(dataset), "examples to", OUTPUT_FILE)