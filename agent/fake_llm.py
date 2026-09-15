"""
fake_llm.py - a scripted stand-in for the real model.

Why it exists: you should be able to watch the loop work without spending
credits, without internet, and with FULL control over what "the model"
says. Tests use it. The offline course chapters use it. When something
goes wrong in the real agent, replaying the same script here tells you
whether the bug is in your loop or in the model's behaviour.

A script is a list of replies. Each reply is either:
  - a string                          -> the model answers with text and stops
  - a list of (tool_name, arguments)  -> the model asks for those tool calls
"""

import json

from .llm import ModelReply


class FakeLLM:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[list[dict]] = []   # every message list we were given, for inspection
        self.counter = 0

    def __call__(self, messages: list[dict], tools: list[dict]) -> ModelReply:
        self.calls.append([dict(message) for message in messages])
        if not self.script:
            return ModelReply(message={"role": "assistant", "content": "(script finished)"}, finish_reason="stop")

        reply = self.script.pop(0)
        if isinstance(reply, str):
            return ModelReply(
                message={"role": "assistant", "content": reply},
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5},
            )

        tool_calls = []
        for name, arguments in reply:
            self.counter += 1
            arguments_json = arguments if isinstance(arguments, str) else json.dumps(arguments)
            tool_calls.append({
                "id": f"call_{self.counter}",
                "type": "function",
                "function": {"name": name, "arguments": arguments_json},
            })
        return ModelReply(
            message={"role": "assistant", "content": None, "tool_calls": tool_calls},
            finish_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
