"""
Chapter 10 - Your own tools. The blueprint in someone else's app.

Nothing in agent/loop.py knows about files. It knows about a ToolRegistry.
That is the whole point of the design: to reuse the agent in another app
(a formulation assistant, a support bot, a data checker) you write NEW
tools and keep everything else.

The recipe, every time:
  1. Write plain functions. Text (or JSON-able values) in, text out.
     Return error messages, do not raise, for anything the model can fix.
  2. Describe each one with a Tool(...): name, description written FOR
     the model, JSON-schema parameters whose names match the function.
  3. Put them in a ToolRegistry and hand it to Agent(...).
  4. Write a system prompt that says what the agent is for.

This chapter builds a tiny formulation helper with two tools and runs it
offline with the fake model. Swap FakeLLM for the real llm function from
agent/__init__.py and it works for real, unchanged.

Run it:
    python3 course/ch10_your_own_tools.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.fake_llm import FakeLLM
from agent.loop import Agent
from agent.tools import Tool, ToolRegistry

# ----- 1. Plain functions: your app's real logic goes here. -----------------------
INGREDIENTS = {
    "glycerin": {"function": "humectant", "typical_percent": "2-5", "note": "sticky above 10%"},
    "niacinamide": {"function": "active", "typical_percent": "2-5", "note": "keep pH 5-7"},
    "phenoxyethanol": {"function": "preservative", "typical_percent": "0.5-1", "note": "max 1%"},
}


def lookup_ingredient(name: str) -> str:
    """Return what we know about an ingredient, or say we do not know it."""
    info = INGREDIENTS.get(name.strip().lower())
    if info is None:
        return f"Unknown ingredient {name!r}. Known: {', '.join(INGREDIENTS)}."
    return f"{name}: {info['function']}, typical {info['typical_percent']}%. Note: {info['note']}"


def check_total(percentages: list[float]) -> str:
    """Add up a list of percentages and say whether they make a valid 100% formula."""
    total = round(sum(percentages), 3)
    if total == 100:
        return "Total is exactly 100%. Valid."
    return f"Total is {total}%, which is {round(100 - total, 3)}% away from 100%."


# ----- 2. Describe them for the model. -------------------------------------------------
tools = [
    Tool(
        name="lookup_ingredient",
        description="Look up the function, typical use level and cautions for a cosmetic ingredient.",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        function=lookup_ingredient,
    ),
    Tool(
        name="check_total",
        description="Check whether a list of ingredient percentages adds up to 100.",
        parameters={"type": "object",
                    "properties": {"percentages": {"type": "array", "items": {"type": "number"}}},
                    "required": ["percentages"]},
        function=check_total,
    ),
]

# ----- 3. Registry + Agent. The loop is untouched. ----------------------------------------
script = [
    [("lookup_ingredient", {"name": "niacinamide"}), ("lookup_ingredient", {"name": "unicorn dust"})],
    [("check_total", {"percentages": [88.5, 4, 3, 1, 3]})],
    "Niacinamide is an active used at 2-5% (keep pH 5-7). I do not know 'unicorn dust'. "
    "Your percentages add up to 99.5%, so add 0.5% water.",
]
agent = Agent(
    llm=FakeLLM(script),
    tools=ToolRegistry(tools),
    system_prompt="You are a formulation assistant. Use the tools; never guess use levels.",
    max_steps=8,
)


def show(event):
    if event["type"] == "tool_call":
        print(f"[act]     {event['name']}({event['arguments']})")
    elif event["type"] == "tool_result":
        print(f"[observe] {event['result']}")


answer = agent.run("Tell me about niacinamide and unicorn dust, then check 88.5, 4, 3, 1, 3.", on_event=show)
print("\n[answer]", answer)
print("\nSame Agent class, same loop, zero file tools. That is the blueprint.")
