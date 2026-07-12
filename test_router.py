#!/usr/bin/env python3

from router.infer_router import classify

examples = [
    "Name the three primary colors in the RGB color model and briefly explain why displays use RGB instead of RYB.",
    "What is the difference between machine learning and deep learning? Briefly explain how each works.",
    "Explain the difference between RAM and ROM in a computer. What is each type used for?",
    "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?",
    "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is needed for 30 cookies? If sugar costs $2.40 per cup, what is the total cost of sugar for 30 cookies?",
    "Classify the sentiment of this customer review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late and the packaging was damaged, but the item worked perfectly and customer support resolved my complaint within an hour.'",
    "Classify the sentiment of this tweet as Positive, Negative, or Neutral and give a one-sentence reason: 'Just got my order. Box was dented and the manual was missing, but honestly the device itself is flawless and set up in under 5 minutes.'",
    "Summarize the following passage in exactly two sentences:\n\nMachine learning is increasingly deployed in healthcare for diagnosis, treatment planning, and patient monitoring. These systems analyse medical images, predict patient deterioration, and spot patterns in electronic health records that might be missed by human clinicians. However, concerns remain about model interpretability, data privacy, liability when errors occur, and the potential for algorithmic bias to worsen existing healthcare disparities. Regulatory frameworks are still catching up with the pace of deployment, creating uncertainty for healthcare providers and technology developers alike.",
    "Summarize the following passage in exactly three bullet points, each no longer than 15 words:\n\nRemote work has transformed how companies operate globally. Employees gain flexibility and reduced commute times, leading to reported improvements in work-life balance. However, challenges persist around collaboration, company culture, and the blurring of personal and professional boundaries. Organisations are responding by investing in digital collaboration tools and rethinking office space as a hub for social and creative work rather than daily attendance.",
    "Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE:\n\nOn March 15 2023, Sundar Pichai announced that Google would open a new AI research lab in Zurich, partnering with ETH Zurich to focus on large language model safety.",
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