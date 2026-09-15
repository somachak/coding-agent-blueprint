"""
tools.py - what a tool is, and the registry that runs them.

A tool is three things kept together in one object:

  1. a Python function that does the work            (function)
  2. a name and a description the model reads        (name, description)
  3. the shape of its arguments, as a JSON Schema    (parameters)

The model never runs Python. It only ever writes TEXT that says
"call this name with these JSON arguments". The registry turns that
text back into a real function call, and turns whatever happens (a result
or an error) back into text for the model. Looking a tool up is not
running it; run() does the running.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON Schema: {"type": "object", "properties": {...}, "required": [...]}
    function: Callable[..., Any]

    def schema(self) -> dict:
        """The exact shape the OpenAI-compatible API expects in the `tools` list."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """A dictionary from tool name to Tool, plus a safe way to run one."""

    def __init__(self, tools: list[Tool] | None = None):
        self.by_name: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self.by_name:
            raise ValueError(f"A tool named {tool.name!r} is already registered.")
        self.by_name[tool.name] = tool

    def names(self) -> list[str]:
        return list(self.by_name.keys())

    def schemas(self) -> list[dict]:
        """The menu we send to the model with every request."""
        return [tool.schema() for tool in self.by_name.values()]

    def run(self, name: str, arguments_json: str) -> str:
        """Run one tool call and ALWAYS return text, never raise.

        Whatever goes wrong becomes a message the model can read and react
        to. If we raised instead, the loop would crash and the model would
        never get the chance to fix its own mistake.
        """
        tool = self.by_name.get(name)
        if tool is None:
            return f"Error: there is no tool called {name!r}. Available tools: {', '.join(self.names())}."

        try:
            arguments = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as error:
            return f"Error: the arguments were not valid JSON ({error}). You sent: {arguments_json[:300]}"

        if not isinstance(arguments, dict):
            return "Error: the arguments must be a JSON object, like {\"path\": \"x.py\"}."

        try:
            result = tool.function(**arguments)
        except TypeError as error:
            # The model sent a wrong argument name, or forgot a required one.
            return f"Error: wrong arguments for {name}: {error}"
        except Exception as error:  # noqa: BLE001 - we want *every* failure to become text
            return f"Error: {name} failed with {type(error).__name__}: {error}"

        if isinstance(result, str):
            return result
        return json.dumps(result)
