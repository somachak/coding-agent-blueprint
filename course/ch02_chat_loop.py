"""
Chapter 2 - The chat loop. Memory is just a list that grows.

Chapter 1 sent two messages and printed one answer. A chat is that same
call inside a `while True` loop, with ONE rule added:

    every message, ours and the model's, gets appended to the same list.

The model has no memory of its own. Each call is brand new. It only
"remembers" because we send the whole list back every single time.

Run it:
    python3 course/ch02_chat_loop.py

Try: say your name, then ask "what is my name?". Then delete the line
marked (A) and ask again. That is what "no memory" looks like.
"""

import json
import os
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for raw_line in (PROJECT_ROOT / ".env").read_text().splitlines():
    line = raw_line.strip()
    if line and not line.startswith("#") and "=" in line:
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ["LLM_MODEL"]


def ask_model(messages: list[dict]) -> dict:
    """Send the whole conversation, return the model's next message (a dict).

    This is Chapter 1 wrapped in a function so the loop below stays short.
    """
    payload = {"model": MODEL, "messages": messages}
    request = urllib.request.Request(
        url=BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.loads(response.read().decode("utf-8"))
    return reply["choices"][0]["message"]


# The conversation starts with only the rules. It will grow.
messages = [{"role": "system", "content": "You are a friendly assistant. Keep answers short."}]

print("Chat with the model. Type 'exit' to stop.\n")
while True:
    user_text = input("You: ").strip()
    if user_text.lower() in {"exit", "quit"}:
        break
    if user_text == "":
        continue

    messages.append({"role": "user", "content": user_text})      # remember what you said

    message = ask_model(messages)                                   # the model sees ALL of it

    messages.append(message)                                        # (A) remember what it said
    print(f"Model: {message['content']}")
    print(f"       (the list now has {len(messages)} messages)\n")
