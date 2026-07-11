#!/usr/bin/env python3

from router.infer_router import classify

examples = [
    "What is the capital of Australia, and what body of water is it near?",
    "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many items remain?",
    "Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.",
    "Summarize the following in exactly one sentence: The global transition to renewable energy faces a critical bottleneck in grid infrastructure, where legacy alternating current (AC) networks struggle to manage the intermittent, highly distributed generation profiles of utility-scale solar and offshore wind farms without causing destabilizing voltage fluctuations; consequently, engineering firms are aggressively deploying high-voltage direct current (HVDC) transmission lines and grid-scale lithium-iron-phosphate (LFP) battery storage systems to act as dynamic buffers, yet this technological overhaul remains severely constrained by regulatory delays, geopolitical conflicts over rare-earth mineral supply chains, and a stark deficit in specialized electrical engineering talent.",
    "Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.",
    "This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.",
    "Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?",
    "Write a Python function that returns the second-largest number in a list, handling duplicates correctly.",
    "An item costs $71. It has a 30% discount applied, then 8% sales tax is added to the discounted price. What is the final price, rounded to 2 decimal places?",
    "What is 424 + 686?",
    "Find and explain the bug in this Python function:\n```python\ndef is_even(n):\n    if n % 2 == 1:\n        return True\n    return False\n```",
    "Summarize this in exactly 2 bullet points, each under 15 words: The city's transit authority reported a 22% rise in ridership after launching a flat monthly pass, though fare revenue per trip fell 9% as riders shifted from single tickets. Maintenance costs also rose due to an aging bus fleet, prompting the authority to request emergency funding from the state to cover a projected budget gap next fiscal year.",
    "Write a Python function `group_anagrams(words)` that groups a list of strings into lists of anagrams of each other. Return a list of groups, each group sorted alphabetically, and the groups sorted by their first element.",
    "Classify the sentiment of this review and justify it in one sentence: \"This laptop's battery easily lasts a full workday, and the screen is gorgeous.\"",
]

print("=" * 80)
print("AMD Router Demo")
print("=" * 80)

for i, prompt in enumerate(examples, 1):
    result = classify(prompt)

    print(f"\nExample {i}")
    print("-" * 80)
    print(f"Prompt      : {prompt}")
    print(f"Category    : {result['category']}")
    print(f"Difficulty  : {result['difficulty']}")
    print(f"Confidence  : {result['confidence']:.4f}")

print("\nDone.")