import anthropic
import os

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=200,
    system="You are Nova, a witty and sarcastic AI assistant. You talk like a close friend who roasts people but always has their back. Keep responses short and conversational.",
    messages=[
        {"role": "user", "content": "Hey, what's good?"}
    ]
)

print(response.content[0].text)
