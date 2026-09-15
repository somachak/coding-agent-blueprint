"""
Chapter 7 - The web window. Same loop, events go to a browser instead.

agent/server.py is a small HTTP server (standard library, no framework).
It has four routes:

    GET  /        the page:  web/index.html
    POST /chat    run one user message; STREAM the events back as they happen
    GET  /state   the whole message list + token counts (the inspector tab)
    POST /reset   forget the conversation

The streaming part is the only new idea. Instead of one JSON reply at the
end, the server keeps the connection open and writes one line per event:

    data: {"type": "tool_call", "name": "read_file", ...}
    (blank line)

That format is called Server-Sent Events (SSE). The page reads the lines
as they arrive and draws a card for each one. Look at handle_chat() in
agent/server.py: the on_event function the loop calls is literally
"write one line to the browser".

This chapter starts that real server with the FAKE model, so you can
click around the page with no key and no cost. Open the URL it prints.
Send any message and watch the Loop tab and the Messages tab.

Run it:
    python3 course/ch07_web_window.py

To see the raw stream the page reads, in a second terminal:
    curl -N -X POST http://127.0.0.1:8765/chat -H 'Content-Type: application/json' -d '{"text": "hi"}'
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Settings
from agent.fake_llm import FakeLLM
from agent.loop import Agent
from agent.server import serve
from agent.tools import ToolRegistry
from agent.workspace import Workspace, build_tools

# Enough scripted replies for three user messages.
script = [
    [("list_files", {})],
    [("write_file", {"path": "notes.md", "content": "# Notes\n\n- the loop just loops\n"})],
    [("read_file", {"path": "notes.md"})],
    "I looked around, wrote notes.md and read it back. (This is a scripted fake model.)",
    [("run_command", {"command": "python3 -c \"print(6*7)\""})],
    "The command printed 42. (Still the fake model.)",
    [("edit_file", {"path": "notes.md", "old_text": "just loops", "new_text": "loops until there are no tool calls"})],
    "Edited notes.md. (Fake model, last scripted reply.)",
]

workspace = Workspace(tempfile.mkdtemp(prefix="ch07-"))
agent = Agent(
    llm=FakeLLM(script),
    tools=ToolRegistry(build_tools(workspace)),
    system_prompt="You are a coding agent (offline demo).",
    max_steps=10,
)
settings = Settings(api_key="", base_url="(offline)", model="fake scripted model", workspace=workspace.root, max_steps=10)

print("Offline demo: the model is scripted, nothing leaves your machine.")
serve(agent, settings)
