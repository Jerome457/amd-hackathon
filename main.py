import os
from openai import OpenAI

import json


def solve(prompt):
    client = OpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY"),
    base_url=os.environ.get("FIREWORKS_BASE_URL")
    )


    MODELS = os.environ["ALLOWED_MODELS"].split(",")

    response = client.chat.completions.create(
        model="accounts/fireworks/models/kimi-k2p7-code",
        messages=[{
            "role": "user",
            "content": prompt,
        }],
    )
    return f"{response.choices[0].message.content}"

def main():
    input_path = "./input/tasks.json"
    output_path = "./output/results.json"

    with open(input_path, "r") as f:
        tasks = json.load(f)

    results = []

    for task in tasks:
        answer = solve(task["prompt"])

        results.append({
            "task_id": task["task_id"],
            "answer": answer
        })

    os.makedirs("./output", exist_ok=True)

    # Write output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()