"""
Chapter 8 - Context: what the model sees first, and what to do when the list gets too long.

Two mechanisms, both about the message list:

1. THE SYSTEM PROMPT is message 0 and is sent on every call. It holds the
   standing rules. If the workspace has an AGENTS.md file, its text is
   appended, so a project can teach the agent its own conventions without
   changing Python. (See build_system_prompt in agent/__init__.py.)

2. COMPACTION. The list only grows, and every model has a window. Before
   each model call the Agent estimates the size; past 80% of the window
   it shrinks the OLD part in two steps: first blank old tool results
   (cheap), then ask the model to summarise the old part into one message
   (one extra call). Recent messages stay verbatim, and the cut never
   separates a tool call from its result. (See agent/context.py.)

This chapter runs offline with the fake model and a tiny window so you can
watch compaction happen and see exactly what the list looks like after.

Run it:
    python3 course/ch08_context_window.py

What breaks if you remove things:
  - no compaction at all       -> after enough steps, HTTP 400 "context length exceeded",
                                  or quietly worse answers long before that
  - cut at a tool message      -> HTTP 400: a tool result without its tool call
  - no AGENTS.md support       -> nothing breaks; you just repeat the rules in every chat
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import build_system_prompt
from agent.context import estimate_tokens
from agent.debug import check_message_order, print_messages
from agent.fake_llm import FakeLLM
from agent.loop import Agent
from agent.tools import ToolRegistry
from agent.workspace import Workspace, build_tools

# ----- 1. The system prompt, with and without AGENTS.md ---------------------------
workspace = Workspace(tempfile.mkdtemp(prefix="ch08-"))
print("System prompt WITHOUT AGENTS.md ends with:")
print("   ...", build_system_prompt(workspace).splitlines()[-1])

workspace.write_file("AGENTS.md", "Always write tests in tests/. Never use print for logging.")
print("System prompt WITH AGENTS.md ends with:")
print("   ...", build_system_prompt(workspace).splitlines()[-1])
print()

# ----- 2. Compaction, watched live ---------------------------------------------------
# A file with a lot of text, read many times, makes the list grow fast.
workspace.write_file("big.txt", "lorem ipsum " * 400)
script = [[("read_file", {"path": "big.txt"})]] * 8 + [
    "Summary requested? Here it is: Goal: read big.txt. Progress: read it 8 times.",  # used by the summariser
    "All done.",
]
fake = FakeLLM(script)
agent = Agent(
    llm=fake,
    tools=ToolRegistry(build_tools(workspace)),
    system_prompt=build_system_prompt(workspace),
    max_steps=20,
    context_window=4000,          # tiny on purpose; real models have 100k+
)


def show(event):
    if event["type"] == "model_call":
        print(f"step {event['step']}: {event['message_count']} messages, ~{estimate_tokens(agent.messages)} tokens")
    if event["type"] == "compaction":
        print(f"   >>> COMPACTION: ~{event['before_tokens']} tokens -> ~{event['after_tokens']} tokens")


agent.run("Read big.txt over and over.", on_event=show)

print("\nThe list after the run:")
print_messages(agent.messages)
print("\nOrder problems:", check_message_order(agent.messages) or "none")
