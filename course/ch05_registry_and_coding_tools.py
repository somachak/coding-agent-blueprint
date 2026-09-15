"""
Chapter 5 - The registry, and the five tools that make it a CODING agent.

Chapter 4 had a lookup table of two functions. Now we make that table the
centre of the design, and we fill it with the five tools every serious
minimal coding agent has (Pi has four of them; Thorsten Ball's has three):

    list_files    look around
    read_file     eyes (with line numbers, so edits can quote exact text)
    write_file    create or replace a whole file
    edit_file     change ONE exact piece of text (the safety feature)
    run_command   run things: python3 app.py, ls, pytest, git ...

Three rules appear here that the loop depends on:
  1. Every path is checked so the agent stays inside its workspace folder.
  2. Every tool returns TEXT, even when it fails. An exception would crash
     the loop; an error message lets the model read it and try again.
  3. Big outputs are cut in the middle. The model only needs the start
     (what happened) and the end (where it failed).

Run it:
    python3 course/ch05_registry_and_coding_tools.py

What breaks if you remove things:
  - the try/except in run_tool        -> one bad argument crashes the whole program
  - safe_path                          -> "read ../../.ssh/id_rsa" would work
  - the uniqueness check in edit_file  -> the model changes the wrong place silently
  - the timeout in run_command         -> `python3 -i` hangs the agent forever
  - truncate_output                    -> one `pip install` log fills the context window
"""

import json
import os
import subprocess
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
WORKSPACE = os.path.realpath(PROJECT_ROOT / "workspace")
MAX_STEPS = 15
MAX_OUTPUT_CHARS = 8000


# ----- Rule 1: every path is checked ---------------------------------------------
def safe_path(relative_path: str) -> str:
    """A real path inside WORKSPACE, or a ValueError. realpath() follows '..' and symlinks."""
    candidate = os.path.realpath(os.path.join(WORKSPACE, relative_path))
    if candidate != WORKSPACE and not candidate.startswith(WORKSPACE + os.sep):
        raise ValueError(f"{relative_path!r} is outside the workspace and was refused.")
    return candidate


# ----- Rule 3: big outputs are cut in the middle ---------------------------------
def truncate_output(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    keep = MAX_OUTPUT_CHARS // 2
    return text[:keep] + f"\n... [{len(text) - MAX_OUTPUT_CHARS} characters cut] ...\n" + text[-keep:]


# ----- The five tools -------------------------------------------------------------
def list_files(path: str = ".") -> str:
    start = safe_path(path)
    lines = []
    for folder, subfolders, files in os.walk(start):
        subfolders[:] = [name for name in sorted(subfolders) if name not in {".git", "__pycache__"}]
        for name in sorted(files):
            full = os.path.join(folder, name)
            lines.append(f"{os.path.relpath(full, WORKSPACE)}  ({os.path.getsize(full)} bytes)")
    return "\n".join(lines) or "(the folder is empty)"


def read_file(path: str) -> str:
    full = safe_path(path)
    if not os.path.isfile(full):
        return f"Error: {path!r} does not exist."
    with open(full, encoding="utf-8") as file:
        lines = file.read().splitlines()
    numbered = [f"{number:5d}| {text}" for number, text in enumerate(lines, start=1)]
    return truncate_output("\n".join(numbered))


def write_file(path: str, content: str) -> str:
    full = safe_path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as file:
        file.write(content)
    return f"Wrote {len(content)} characters to {path}."


def edit_file(path: str, old_text: str, new_text: str) -> str:
    full = safe_path(path)
    if not os.path.isfile(full):
        return f"Error: {path!r} does not exist. Use write_file to create it."
    with open(full, encoding="utf-8") as file:
        original = file.read()
    count = original.count(old_text)
    if count == 0:
        return "Error: old_text was not found. Read the file and copy the text exactly."
    if count > 1:
        return f"Error: old_text appears {count} times. Include more surrounding lines so it is unique."
    with open(full, "w", encoding="utf-8") as file:
        file.write(original.replace(old_text, new_text, 1))
    return f"Edited {path}."


def run_command(command: str, timeout_seconds: int = 60) -> str:
    try:
        done = subprocess.run(command, shell=True, cwd=WORKSPACE, capture_output=True,
                              text=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        return f"Error: the command did not finish within {timeout_seconds} seconds and was killed."
    output = done.stdout + ("\n[stderr]\n" + done.stderr if done.stderr else "")
    return truncate_output(f"exit code {done.returncode}\n{output or '(no output)'}")


# ----- The registry: ONE table, both halves side by side --------------------------
# Left half is what the model reads (the schema). Right half is what we run.
# If a name is in one half but not the other, the agent breaks: try it.
def schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}}

REGISTRY = {
    "list_files": {
        "schema": schema("list_files", "List every file in the workspace, with sizes. Start here.",
                         {"path": {"type": "string", "description": "Folder, default '.'"}}, []),
        "function": list_files,
    },
    "read_file": {
        "schema": schema("read_file", "Read a text file with line numbers.",
                         {"path": {"type": "string"}}, ["path"]),
        "function": read_file,
    },
    "write_file": {
        "schema": schema("write_file", "Create a file or completely replace its content.",
                         {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
        "function": write_file,
    },
    "edit_file": {
        "schema": schema("edit_file", "Replace one exact occurrence of old_text with new_text. old_text must be unique.",
                         {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                         ["path", "old_text", "new_text"]),
        "function": edit_file,
    },
    "run_command": {
        "schema": schema("run_command", "Run a shell command in the workspace and return its output and exit code.",
                         {"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}}, ["command"]),
        "function": run_command,
    },
}

TOOLS = [entry["schema"] for entry in REGISTRY.values()]


# ----- Rule 2: every tool call returns text, never an exception -------------------
def run_tool(name: str, arguments_json: str) -> str:
    entry = REGISTRY.get(name)
    if entry is None:
        return f"Error: no tool called {name!r}. Available: {', '.join(REGISTRY)}."
    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as error:
        return f"Error: arguments were not valid JSON ({error})."
    try:
        return entry["function"](**arguments)
    except TypeError as error:
        return f"Error: wrong arguments for {name}: {error}"
    except Exception as error:  # noqa: BLE001
        return f"Error: {name} failed with {type(error).__name__}: {error}"


def ask_model(messages: list[dict]) -> dict:
    payload = {"model": MODEL, "messages": messages, "tools": TOOLS}
    request = urllib.request.Request(
        url=BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        reply = json.loads(response.read().decode("utf-8"))
    return reply["choices"][0]["message"]


# ----- The loop: unchanged from Chapter 4 except run_tool() -----------------------
os.makedirs(WORKSPACE, exist_ok=True)
messages = [
    {"role": "system", "content": (
        "You are a careful coding agent working inside one workspace folder. "
        "Look before you touch. After changing code, run it to check. "
        "Finish with a short summary of what you did."
    )},
    {"role": "user", "content": "Create countdown.py that prints 5 down to 1 then 'Go!'. Run it. "
                                "Then change it to start from 3 and run it again."},
]

for step in range(1, MAX_STEPS + 1):
    print(f"--- step {step} ({len(messages)} messages) ---")
    message = ask_model(messages)
    messages.append(message)
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        print("\nFinal answer:\n" + (message["content"] or ""))
        break
    for call in tool_calls:
        name = call["function"]["name"]
        arguments_json = call["function"]["arguments"]
        result = run_tool(name, arguments_json)        # <- never raises
        print(f"    {name}({arguments_json[:70]}) -> {result[:70]!r}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
else:
    print(f"\nStopped after {MAX_STEPS} steps.")
