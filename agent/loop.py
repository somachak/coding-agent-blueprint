"""
loop.py - the agent loop. This is the whole idea of an "agent".

    1. REASON   send the conversation to the model
    2. ACT      if it asked for tools, run them
    3. OBSERVE  append each result to the conversation
    4. repeat, until the model answers with text instead of a tool call

Everything else in the package exists to feed this loop or to show you
what it is doing. It has no idea how HTTP works (llm.py) or what a file
is (workspace.py). It only knows: messages go in, a message comes out,
tool calls get run, results get appended.

Events
------
The loop reports what it is doing by calling `on_event(event)` with a
plain dictionary. The terminal prints them, the web page streams them,
your own app can log them. The loop itself never prints. The event types:

    user          the message that started this turn
    compaction    old messages were shrunk to fit the window
    model_call    about to ask the model (step number, message count)
    model_reply   the model answered (did it ask for tools? how many?)
    tool_call     one tool is about to run (name, arguments as JSON text)
    tool_result   what that tool returned (text)
    answer        the final text; the turn is over
    error         something failed; the text says what and what was done

A turn either finishes or is rolled back
----------------------------------------
If the model call fails (bad key, bad model name, network), the messages
added during this turn are removed again, so the list never holds a
half-finished turn. The next message starts from a clean state. Work the
tools already did on disk is NOT undone; the error event says so.
"""

from typing import Callable

from .context import estimate_tokens, fit_context
from .llm import LLMError, ModelReply
from .tools import ToolRegistry

Event = dict
EventHandler = Callable[[Event], None]
LLMFunction = Callable[[list[dict], list[dict]], ModelReply]

# More than this many tool calls in ONE reply is a model misbehaving, not a plan.
MAX_TOOL_CALLS_PER_REPLY = 20


def no_op(event: Event) -> None:
    """The default event handler: do nothing."""


class Agent:
    def __init__(
        self,
        llm: LLMFunction,
        tools: ToolRegistry,
        system_prompt: str,
        max_steps: int = 25,
        context_window: int = 0,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.context_window = context_window   # 0 means "never compact"
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def run(self, user_text: str, on_event: EventHandler = no_op) -> str:
        """Handle ONE user message all the way to a final answer.

        Returns the answer text. Everything that happened on the way is
        reported through on_event.
        """
        user_message = {"role": "user", "content": user_text}
        self.messages.append(user_message)
        on_event({"type": "user", "text": user_text})

        for step in range(1, self.max_steps + 1):
            # ---- 0. Make sure the conversation still fits the window --------
            self._fit_context(on_event)

            # ---- 1. REASON --------------------------------------------------
            on_event({"type": "model_call", "step": step, "message_count": len(self.messages)})
            try:
                reply = self.llm(self.messages, self.tools.schemas())
            except LLMError as error:
                removed = self._roll_back(user_message)
                on_event({
                    "type": "error",
                    "step": step,
                    "text": f"{error}\nThis message was removed from the conversation ({removed} messages rolled back); "
                            "files the tools already changed stay changed.",
                })
                return f"Error: {error}"

            self._add_usage(reply.usage)
            self.messages.append(reply.message)
            tool_calls = reply.message.get("tool_calls", [])
            on_event({
                "type": "model_reply",
                "step": step,
                "finish_reason": reply.finish_reason,
                "tool_call_count": len(tool_calls),
                "text": reply.message.get("content") or "",
                "usage": reply.usage,
            })

            # ---- No tool calls means the model has finished. ----------------
            if not tool_calls:
                answer = reply.message.get("content") or ""
                on_event({"type": "answer", "step": step, "text": answer})
                return answer

            # ---- 2. ACT and 3. OBSERVE, once per requested call ------------
            cut_off = reply.finish_reason == "length"   # the reply hit the output limit
            for number, call in enumerate(tool_calls, start=1):
                name = call["function"]["name"]
                arguments_json = call["function"]["arguments"]
                on_event({"type": "tool_call", "step": step, "id": call["id"], "name": name, "arguments": arguments_json})

                if cut_off:
                    result = ("Error: your reply was cut off by the output limit, so these arguments may be "
                              "incomplete. The tool was not run. Re-issue the call with complete arguments.")
                elif number > MAX_TOOL_CALLS_PER_REPLY:
                    result = (f"Error: more than {MAX_TOOL_CALLS_PER_REPLY} tool calls in one reply. "
                              "This call was not run. Ask for fewer calls at a time.")
                else:
                    result = self.tools.run(name, arguments_json)

                # The API insists: every tool call gets exactly one tool
                # message, linked by tool_call_id, before the next model call.
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
                on_event({"type": "tool_result", "step": step, "id": call["id"], "name": name, "result": result})

            # Loop round: the model now sees the results and decides again.

        # The brake: too many steps for one message. Stop and say so.
        text = f"Stopped after {self.max_steps} steps without a final answer."
        on_event({"type": "error", "step": self.max_steps, "text": text})
        self.messages.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        """Forget the conversation but keep the system prompt."""
        self.messages = self.messages[:1]
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def _roll_back(self, user_message: dict) -> int:
        """Remove this turn's messages. Returns how many were removed.

        We look for the exact user message object that started the turn;
        everything from it to the end belongs to this turn. If compaction
        summarised it away already, we add a short assistant note instead,
        so the list stays well formed either way.
        """
        for index, message in enumerate(self.messages):
            if message is user_message:
                removed = len(self.messages) - index
                del self.messages[index:]
                return removed
        self.messages.append({"role": "assistant", "content": "The previous request failed and was abandoned."})
        return 0

    def _fit_context(self, on_event: EventHandler) -> None:
        """Shrink old messages when the estimate passes 80% of the window.

        A failure here (the summariser is a model call too) is reported as
        an error event and the turn continues with the unshrunk list.
        """
        if self.context_window <= 0:
            return
        budget = int(self.context_window * 0.8)
        before = estimate_tokens(self.messages)
        if before <= budget:
            return
        try:
            self.messages = fit_context(self.messages, budget, self.llm)
        except LLMError as error:
            on_event({"type": "error", "text": f"Compaction failed, continuing without it: {error}"})
            return
        after = estimate_tokens(self.messages)
        if after < before:
            on_event({"type": "compaction", "before_tokens": before, "after_tokens": after})

    def _add_usage(self, usage: dict) -> None:
        for key in self.total_usage:
            self.total_usage[key] += int(usage.get(key, 0) or 0)
