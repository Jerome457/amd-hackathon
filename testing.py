import os
import json
import sys

from main import get_response_with_source

INPUT_PATH = "./input/tasks.json" # remove the . for docker test run???
OUTPUT_DIR = "./output"
OUTPUT_PATH = OUTPUT_DIR + "/results.json"

def get_tasks():
    tasks = None
    with open(INPUT_PATH, "r") as input_file:
        tasks = json.load(input_file)
    return tasks

def write_output(answers):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        json.load(f)

def main():
    tasks = get_tasks()
    answers = []
    local_llm_count = 0
    fireworks_count = 0
    total_tasks = len(tasks)

    for index, task in enumerate(tasks, start=1):
        answer, source = get_response_with_source(task["prompt"])

        if source == "local_llm":
            local_llm_count += 1
        else:
            fireworks_count += 1

        answers.append(
            {
                "task_id": task["task_id"],
                "answer": answer,
            }
        )

        if index % 5 == 0 or index == total_tasks:
            print(f"Processed {index}/{total_tasks} tasks...", flush=True)

    write_output(answers)
    print(f"Results file generated at {OUTPUT_PATH}")
    print(f"local_llm used: {local_llm_count}")
    print(f"fireworks used: {fireworks_count}")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
