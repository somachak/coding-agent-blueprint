"""
Chapter 4 - The loop. Repeat until the model stops asking for tools.

Chapter 3 did one round trip by hand: call, run the tool, call again.
Real tasks need several: "list the files, then read the biggest one,
then tell me what it does" is three tool calls before an answer. Nobody
knows in advance how many. So we wrap the round trip in `while True`
and add ONE stopping rule:

    if the reply has no tool_calls, it is the answer -> stop.

And one brake: a maximum number of steps, because a model can get stuck
asking for the same thing forever, and every step costs money.

Run it:
    python3 course/ch04_the_loop.py

What breaks if you remove things:
  - remove the `break` when there are no tool calls  -> the loop asks the
    model again with nothing new; it repeats itself until the brake stops it
  - remove MAX_STEPS                                 -> a stuck model runs
    (and bills you) forever
  - forget `messages.append(message)` before the tool messages -> HTTP 400
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
WORKSPACE = PROJECT_ROOT / "workspace"
MAX_STEPS = 10


# ----- Two tools now. Same shape as Chapter 3, one more function. ---------------
def read_file(path: str) -> str:
    full_path = WORKSPACE / path
    if not full_path.is_file():
        return f"Error: {path} does not exist."
    return full_path.read_text(encoding="utf-8")


def list_files() -> str:
    names = sorted(p.name for p in WORKSPACE.iterdir() if p.is_file())
    if not names:
        return "(the folder is empty)"
    return "\n".join(names)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File name"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the workspace folder.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# A tiny lookup table: the NAME the model uses -> the FUNCTION we run.
# Looking a name up here does not run anything. The () later does.
FUNCTIONS = {
    "read_file": read_file,
    "list_files": list_files,
}


def ask_model(messages: list[dict]) -> dict:
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS}
    request = urllib.request.Request(
        url=BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        reply = json.loads(response.read().decode("utf-8"))
    return reply["choices"][0]["message"]


# ----- Set the scene: three files, one of them clearly the biggest. ---------------
WORKSPACE.mkdir(exist_ok=True)
(WORKSPACE / "small.txt").write_text("tiny\n")
(WORKSPACE / "medium.txt").write_text("a few words here\n")
(WORKSPACE / "big.txt").write_text("This file explains the project: it is a recipe book for skincare formulas.\n" * 3)

messages = [
    {"role": "system", "content": "You are an assistant with tools. Use them instead of guessing."},
    {"role": "user", "content": "Which file in the workspace is the biggest, and what is it about? Read it to find out."},
]

# ----- THE LOOP -----------------------------------------------------------------
for step in range(1, MAX_STEPS + 1):
    print(f"--- step {step}: asking the model ({len(messages)} messages) ---")
    message = ask_model(messages)          # 1. REASON
    messages.append(message)

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:                     # no request = the answer. Stop.
        print("\nFinal answer:")
        print(message["content"])
        break

    for call in tool_calls:                # 2. ACT, once per requested call
        name = call["function"]["name"]
        arguments = json.loads(call["function"]["arguments"])
        function = FUNCTIONS[name]         # look up the function by name
        result = function(**arguments)     # run it with the model's arguments
        print(f"    {name}({arguments}) -> {result[:60]!r}")

        messages.append({                  # 3. OBSERVE: the result goes back in
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result,
        })
    # ...and round we go: the model sees the results and decides again.
else:
    # Python runs this `else` only if the loop never hit `break`.
    print(f"\nStopped after {MAX_STEPS} steps without an answer.")

print("\nRoles in order:", [m["role"] for m in messages])
