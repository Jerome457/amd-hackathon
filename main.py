import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("FIREWORKS_API_KEY"),
    base_url=os.environ.get("FIREWORKS_BASE_URL")
)


MODELS = os.environ["ALLOWED_MODELS"].split(",")

response = client.chat.completions.create(
    model="accounts/fireworks/models/kimi-k2p7-code",
    messages=[{
        "role": "user",
        "content": "Say hello in Spanish",
    }],
)

print(response.choices[0].message.content)