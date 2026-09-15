"""
Chapter 3 - The first tool. This is the chapter where it becomes an agent.

A tool is a normal Python function. The model cannot run it. What the
model CAN do is write a small piece of JSON that says "please call this
function with these arguments". We run the function, put the result back
into the conversation as a "tool" message, and ask the model again.

That round trip is the entire trick:

    model -> "call read_file(path='notes.txt')"      (a tool_call)
    us    -> run it, get text
    us    -> append {"role": "tool", "content": text}
    model -> now answers using what it read

Run it:
    python3 course/ch03_first_tool.py

What breaks if you remove things:
  - leave out `tools` in the payload   -> the model answers from imagination; it
                                          cannot ask for a tool it was never offered
  - forget to append the assistant message with tool_calls before the tool
    message                            -> HTTP 400: the API refuses a tool result
                                          that answers no tool call
  - use the wrong tool_call_id         -> HTTP 400, same reason
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


# ----- 1. The tool itself: a plain function. Text in, text out. ------------------
def read_file(path: str) -> str:
    """Return the contents of a file in the workspace, or an error message."""
    full_path = WORKSPACE / path
    if not full_path.is_file():
        return f"Error: {path} does not exist."
    return full_path.read_text(encoding="utf-8")


# ----- 2. The description the model reads. This is the ONLY thing it sees. -------
# It never sees the Python above. It sees this JSON "menu card".
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File name, e.g. 'notes.txt'"},
                },
                "required": ["path"],
            },
        },
    }
]


def ask_model(messages: list[dict]) -> dict:
    """Same as Chapter 2, plus the tools menu in the payload."""
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


# ----- 3. Set the scene: a file the model cannot know about. --------------------
WORKSPACE.mkdir(exist_ok=True)
(WORKSPACE / "notes.txt").write_text("The launch code is PINEAPPLE-42.\n")

messages = [
    {"role": "system", "content": "You are an assistant with tools. Use them instead of guessing."},
    {"role": "user", "content": "What is the launch code written in notes.txt?"},
]

# ----- 4. First call: the model should ASK for the tool, not answer. --------------
message = ask_model(messages)
messages.append(message)                    # save its request exactly as sent

print("First reply from the model:")
print(json.dumps(message, indent=2))
print()

tool_calls = message.get("tool_calls") or []
if not tool_calls:
    print("The model answered without a tool. It guessed. Some models do this; try another model.")
    raise SystemExit

# ----- 5. Run what it asked for, and put the result back as a tool message. -------
for call in tool_calls:
    name = call["function"]["name"]                       # "read_file"
    arguments_json = call["function"]["arguments"]         # '{"path": "notes.txt"}'  <- TEXT
    arguments = json.loads(arguments_json)                # {"path": "notes.txt"}     <- dict

    print(f"The model asked for: {name}({arguments})")
    result = read_file(**arguments)                       # ** unpacks the dict into keyword arguments
    print(f"We ran it and got:   {result!r}\n")

    messages.append({
        "role": "tool",
        "tool_call_id": call["id"],                       # links this result to that request
        "content": result,
    })

# ----- 6. Second call: now the model has the result and can answer. ---------------
message = ask_model(messages)
messages.append(message)

print("Final answer:")
print(message["content"])
print()
print("Roles in the conversation, in order:", [m["role"] for m in messages])
