"""
Chapter 9 - The troubleshooting lab. Break it on purpose, see what happens.

Every case below reproduces one real failure with the FAKE model, prints
what you would see, why it happens, and what the blueprint does about it.
Run the whole file, or read one case at a time. No key, no cost.

Run it:
    python3 course/ch09_troubleshooting_lab.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.debug import check_message_order
from agent.fake_llm import FakeLLM
from agent.llm import LLMError, clean_assistant_message
from agent.loop import Agent
from agent.tools import ToolRegistry
from agent.workspace import Workspace, build_tools, truncate_output


def fresh_agent(script, max_steps=6, context_window=0):
    workspace = Workspace(tempfile.mkdtemp(prefix="ch09-"))
    registry = ToolRegistry(build_tools(workspace))
    return Agent(llm=FakeLLM(script), tools=registry, system_prompt="sys", max_steps=max_steps,
                 context_window=context_window), workspace


def case(number, title):
    print(f"\n{'=' * 78}\nCASE {number}: {title}\n{'=' * 78}")


def tool_results(agent, on_events):
    return [e["result"] for e in on_events if e["type"] == "tool_result"]


# ---------------------------------------------------------------------------------
case(1, "The model asks for a tool that does not exist")
agent, _ = fresh_agent([[("search_web", {"query": "x"})], "ok"])
events = []
agent.run("go", on_event=events.append)
print("What the model got back:", tool_results(agent, events)[0])
print("Why: the name in the model's request is looked up in the registry; a miss is a normal text result.")
print("Fix: check the schema list and the registry agree (the same dict feeds both in the blueprint).")

# ---------------------------------------------------------------------------------
case(2, "The arguments are not valid JSON (often: the reply was cut off by max_tokens)")
agent, _ = fresh_agent([[("read_file", '{"path": "a.txt')], "ok"])
events = []
agent.run("go", on_event=events.append)
print("What the model got back:", tool_results(agent, events)[0][:90])
print("Why: arguments arrive as TEXT; json.loads failed; the registry turned that into an error message.")
print("Fix: nothing to fix in the loop. If it keeps happening, check finish_reason == 'length' and raise max_tokens.")

# ---------------------------------------------------------------------------------
case(3, "Wrong argument names (the model invented a parameter)")
agent, _ = fresh_agent([[("read_file", {"filename": "a.txt"})], "ok"])
events = []
agent.run("go", on_event=events.append)
print("What the model got back:", tool_results(agent, events)[0])
print("Why: read_file(**{'filename': ...}) raises TypeError; the registry catches it and reports it.")
print("Fix: make the schema's property names match the function's parameter names exactly.")

# ---------------------------------------------------------------------------------
case(4, "A path outside the workspace")
agent, _ = fresh_agent([[("read_file", {"path": "../../../etc/passwd"})], "ok"])
events = []
agent.run("go", on_event=events.append)
print("What the model got back:", tool_results(agent, events)[0])
print("Why: safe_path() resolves '..' and symlinks with realpath() and refuses anything outside the root.")

# ---------------------------------------------------------------------------------
case(5, "edit_file cannot find the text, or finds it twice")
agent, workspace = fresh_agent([
    [("edit_file", {"path": "a.txt", "old_text": "zzz", "new_text": "y"})],
    [("edit_file", {"path": "a.txt", "old_text": "one", "new_text": "1"})],
    "ok",
])
workspace.write_file("a.txt", "one one two")
events = []
agent.run("go", on_event=events.append)
for result in tool_results(agent, events):
    print("What the model got back:", result)
print("Why: exact-and-unique matching is the safety feature; the error tells the model what to do next.")

# ---------------------------------------------------------------------------------
case(6, "The loop never ends (the model keeps calling tools)")
agent, _ = fresh_agent([[("list_files", {})]] * 50, max_steps=4)
answer = agent.run("go")
print("What you see:", answer)
print("Why: max_steps is the brake. Without it a stuck model runs, and bills you, forever.")

# ---------------------------------------------------------------------------------
case(7, "A tool returns a huge output")
big = "x" * 50_000
print("50,000 characters became:", len(truncate_output(big)), "characters, with the middle marked as cut.")
print("Why: the model only needs the start and the end; the rest wastes the context window.")

# ---------------------------------------------------------------------------------
case(8, "A command hangs")
agent, _ = fresh_agent([[("run_command", {"command": "sleep 30", "timeout_seconds": 1})], "ok"])
events = []
agent.run("go", on_event=events.append)
print("What the model got back:", tool_results(agent, events)[0])
print("Why: subprocess.run(timeout=...) kills it. Without a timeout, `python3 -i` would block the agent forever.")

# ---------------------------------------------------------------------------------
case(9, "The API call itself fails (401 / 402 / 429 / model not found)")
def broken_llm(messages, tools):
    raise LLMError("HTTP 401: The API key was refused. Check LLM_API_KEY in .env.")
agent = Agent(llm=broken_llm, tools=ToolRegistry(), system_prompt="sys")
print("What you see as the answer:", agent.run("hi"))
print("Why: llm.chat() turns HTTP errors into LLMError with a hint; the loop reports it instead of crashing.")

# ---------------------------------------------------------------------------------
case(10, "The model answers in prose instead of calling a tool")
agent, _ = fresh_agent(["The file probably contains a greeting."])
print("What you see:", agent.run("What is in hello.py?"))
print("Why: the model was not offered tools, does not support them, or the description did not persuade it.")
print("Fix: confirm `tools` is sent; use a model with tool support; write descriptions that say WHEN to use the tool.")

# ---------------------------------------------------------------------------------
case(11, "HTTP 400 from a broken message order (found locally before sending)")
bad = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "go"},
    {"role": "tool", "tool_call_id": "call_9", "content": "result"},           # no assistant asked for it
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                             "function": {"name": "list_files", "arguments": "{}"}}]},
    {"role": "user", "content": "and now?"},                                    # call_1 never answered
]
for problem in check_message_order(bad):
    print("Problem:", problem)
print("Why: the API demands every tool message answers a tool call that came right before it.")
print("Fix: run check_message_order() on your list when you get a 400; it names the bad index.")

# ---------------------------------------------------------------------------------
case(12, "The assistant message has content: null")
raw = {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
                                                              "function": {"name": "x", "arguments": "{}"}}],
       "reasoning": "private thoughts", "refusal": None}
print("Provider sent:", sorted(raw.keys()))
print("We send back:  ", sorted(clean_assistant_message(raw).keys()))
print("Why: null content is normal next to tool_calls, but extra provider fields can break the NEXT provider.")
print("Fix: clean_assistant_message() keeps only role, content and tool_calls.")

print("\nAll twelve cases ran. Every failure became text or a clean stop; nothing crashed.")
