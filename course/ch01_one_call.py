"""
Chapter 1 - One call to the model.

Before there is an agent, there is a single HTTP request. This file makes
exactly one, and prints what comes back. Everything later in the course
is this same request, repeated, with more things in the message list.

Run it:
    python3 course/ch01_one_call.py

What you should see: one sentence from the model, then the token counts.

What breaks if you remove things:
  - no Authorization header  -> HTTP 401 (the server does not know who you are)
  - wrong LLM_MODEL          -> HTTP 400 or 404 (the server has no such model)
  - messages is empty        -> HTTP 400 (nothing to answer)
"""

import json
import os
import urllib.request
from pathlib import Path

# ----- 1. Settings. A .env file is just NAME=value lines; we read it by hand. ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

for raw_line in (PROJECT_ROOT / ".env").read_text().splitlines():
    line = raw_line.strip()
    if line and not line.startswith("#") and "=" in line:
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip())

API_KEY = os.environ["LLM_API_KEY"]                                    # the secret
BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")  # the server
MODEL = os.environ["LLM_MODEL"]                                        # the model id

# ----- 2. The conversation so far: a list of dictionaries. ----------------------
# Each dictionary has a role (who is speaking) and content (what they said).
messages = [
    {"role": "system", "content": "You are a helpful assistant. Answer in one sentence."},
    {"role": "user", "content": "In plain words, what does a coding agent do?"},
]

# ----- 3. The request. JSON text goes out, JSON text comes back. ------------------
payload = {"model": MODEL, "messages": messages}

request = urllib.request.Request(
    url=BASE_URL + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),   # Python dict -> JSON text -> bytes
    headers={
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json",
    },
    method="POST",
)

with urllib.request.urlopen(request, timeout=60) as response:
    reply = json.loads(response.read().decode("utf-8"))   # bytes -> JSON text -> Python dict

# ----- 4. Find the answer inside the reply. --------------------------------------
# reply["choices"] is a list (usually one item). Each item has a "message",
# which is the same shape as the dictionaries we sent: role + content.
message = reply["choices"][0]["message"]

print("The model said:")
print(message["content"])
print()
print("Tokens used:", reply["usage"])
