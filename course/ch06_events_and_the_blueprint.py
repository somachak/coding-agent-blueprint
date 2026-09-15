"""
Chapter 6 - Events, and the moment the script becomes a reusable blueprint.

Chapters 1-5 were one growing script with print() calls inside the loop.
That is fine for a terminal and useless for anything else: a web page, a
log file, or your own app cannot "see" print(). So the loop stops
printing and starts REPORTING: every time something happens it calls a
function you gave it, with a small dictionary describing the event.

    loop  --on_event({...})-->  terminal printer  /  web page  /  your app

That single change is what turns the script into the package in agent/:

    agent/loop.py       the loop, now a class called Agent, reports events
    agent/tools.py      Tool + ToolRegistry (Chapter 5's table, tidied)
    agent/workspace.py  the five tools (Chapter 5's functions, tidied)
    agent/llm.py        ask_model() with proper error messages
    agent/config.py     the .env reader
    agent/fake_llm.py   a scripted model so you can run all of this OFFLINE

This chapter runs the real package with the FAKE model: no key, no cost,
no internet, and total control over what "the model" says. Watch the
events come out one by one, then look at the finished message list.

Run it:
    python3 course/ch06_events_and_the_blueprint.py

To swap in the real model, replace FakeLLM(...) with the two-line `llm`
function you can see in agent/__init__.py (create_agent). Nothing else changes.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.fake_llm import FakeLLM
from agent.loop import Agent
from agent.tools import ToolRegistry
from agent.workspace import Workspace, build_tools

# ----- 1. A scripted model. Each item is what the model "says" on that call. ------
script = [
    [("list_files", {})],                                            # call 1: look around
    [("write_file", {"path": "hello.py", "content": "print('hi')\n"})],   # call 2: create a file
    [("run_command", {"command": "python3 hello.py"})],              # call 3: run it
    "I created hello.py and ran it. It printed: hi",                 # call 4: the answer
]
fake_model = FakeLLM(script)

# ----- 2. The same wiring create_agent() does, written out in full. ---------------
workspace = Workspace(tempfile.mkdtemp(prefix="ch06-"))
registry = ToolRegistry(build_tools(workspace))
agent = Agent(llm=fake_model, tools=registry, system_prompt="You are a coding agent.", max_steps=10)


# ----- 3. An event handler. The loop calls this; it decides how to show things. ----
def show(event: dict) -> None:
    kind = event["type"]
    if kind == "model_call":
        print(f"[reason]  step {event['step']}: model is looking at {event['message_count']} messages")
    elif kind == "model_reply":
        print(f"[reason]  step {event['step']}: model asked for {event['tool_call_count']} tool call(s)")
    elif kind == "tool_call":
        print(f"[act]     {event['name']}({event['arguments']})")
    elif kind == "tool_result":
        print(f"[observe] {event['result'][:60]!r}")
    elif kind == "answer":
        print(f"[answer]  {event['text']}")
    else:
        print(f"[{kind}]   {event}")


answer = agent.run("Create hello.py that prints hi, and run it.", on_event=show)

# ----- 4. What the model would see next time: the whole list, in order. -----------
print("\nThe conversation the model sees, message by message:")
for index, message in enumerate(agent.messages):
    role = message["role"]
    if role == "assistant" and message.get("tool_calls"):
        calls = ", ".join(c["function"]["name"] for c in message["tool_calls"])
        print(f"  {index}. {role:9s} -> asked for: {calls}")
    else:
        text = (message.get("content") or "")[:60].replace("\n", " ")
        extra = f" (answers {message['tool_call_id']})" if role == "tool" else ""
        print(f"  {index}. {role:9s}{extra}: {text!r}")

print(f"\nThe fake model was called {len(fake_model.calls)} times. Workspace: {workspace.root}")
