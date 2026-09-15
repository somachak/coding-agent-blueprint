"""
debug.py - two helpers for when the agent misbehaves.

    print_messages(messages)        the conversation, one line per message
    check_message_order(messages)   the rules the API enforces, checked locally

Nine times out of ten an HTTP 400 from the model API means the message
list broke one of these rules. Checking them yourself, before sending,
turns a vague server error into a sentence that names the bad index.
"""


def print_messages(messages: list[dict]) -> None:
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            calls = ", ".join(f"{c['function']['name']}[{c['id']}]" for c in message["tool_calls"])
            print(f"{index:3d}. {role:9s} -> {calls}")
        elif role == "tool":
            text = message.get("content", "")[:70].replace("\n", " ")
            print(f"{index:3d}. {role:9s} (answers {message.get('tool_call_id')}): {text!r}")
        else:
            text = (message.get("content") or "")[:70].replace("\n", " ")
            print(f"{index:3d}. {role:9s}: {text!r}")


def check_message_order(messages: list[dict]) -> list[str]:
    """Return a list of problems (empty means the list is well formed).

    The rules:
      1. The first message is the system message.
      2. A tool message must directly follow the assistant message that
         asked for it (or another tool message from the same batch).
      3. Every tool_call_id the assistant asked for gets exactly one tool
         message, and no tool message answers an id that was never asked.
      4. An assistant message either has text content or tool_calls (or both).
    """
    problems = []
    if not messages or messages[0]["role"] != "system":
        problems.append("message 0 should be the system message")

    pending_ids: list[str] = []      # tool call ids still waiting for a tool message
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "assistant":
            if pending_ids:
                problems.append(f"message {index}: assistant spoke while tool calls {pending_ids} were still unanswered")
            pending_ids = [call["id"] for call in message.get("tool_calls", [])]
            if not message.get("tool_calls") and not message.get("content"):
                problems.append(f"message {index}: assistant message has neither content nor tool_calls")
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id in pending_ids:
                pending_ids.remove(call_id)
            else:
                problems.append(f"message {index}: tool message answers {call_id!r}, which no assistant message asked for")
        elif role == "user":
            if pending_ids:
                problems.append(f"message {index}: user spoke while tool calls {pending_ids} were still unanswered")
            pending_ids = []
    if pending_ids:
        problems.append(f"the list ends with unanswered tool calls {pending_ids}")
    return problems
