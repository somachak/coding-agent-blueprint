"""
context.py - keeping the conversation inside the model's window.

The message list only ever grows. Every model has a limit on how much it
can read at once (its "context window"), and quality drops well before
the hard limit. So, before each model call, we check the size and, when
it gets big, we shrink the OLD part of the list in two increasingly
drastic steps:

  1. trim   : replace old tool results with a short placeholder. Cheap,
              no model call. Tool output is consumed once; the model's
              conclusions live on in its later messages.
  2. summarise : ask the model to summarise everything old into one
              message, and keep only the recent messages verbatim.

One rule must never be broken: a tool message must stay right after the
assistant message that asked for it. So we only ever cut the list at a
user or assistant message, never at a tool message.
"""

import json

# A rough but honest estimate: about four characters per token for English
# and code. Good enough to decide WHEN to shrink; not good enough to bill.
CHARS_PER_TOKEN = 4

SUMMARY_INSTRUCTIONS = (
    "Summarise the conversation below for an AI coding agent that will continue the work. "
    "Use these headings: Goal, Progress so far, Key decisions, Files touched, Next steps. "
    "Keep exact file names, function names and error messages. Be brief."
)


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count for a message list."""
    text = json.dumps(messages)
    return len(text) // CHARS_PER_TOKEN


def find_cut_index(messages: list[dict], keep_recent: int) -> int:
    """Index where 'old' ends and 'recent' begins.

    Counts back `keep_recent` messages from the end, then moves the cut
    forward until it lands on a user or assistant message, so an
    assistant tool request is never separated from its tool results.
    Index 0 is the system message; it is never part of the cut.
    """
    cut = max(1, len(messages) - keep_recent)
    while cut < len(messages) and messages[cut]["role"] == "tool":
        cut += 1
    return cut


def trim_old_tool_results(messages: list[dict], keep_recent: int = 12) -> list[dict]:
    """Step 1: blank out tool results older than the recent tail."""
    cut = find_cut_index(messages, keep_recent)
    trimmed = []
    for index, message in enumerate(messages):
        is_old_tool_result = index < cut and message["role"] == "tool"
        if is_old_tool_result and not message["content"].startswith("[old tool output"):
            message = {**message, "content": "[old tool output removed to save space]"}
        trimmed.append(message)
    return trimmed


def summarise_older_messages(messages: list[dict], llm, keep_recent: int = 12) -> list[dict]:
    """Step 2: replace everything old with one summary message from the model."""
    cut = find_cut_index(messages, keep_recent)
    old = messages[1:cut]
    if not old:
        return messages
    transcript = render_transcript(old)
    reply = llm(
        [
            {"role": "system", "content": SUMMARY_INSTRUCTIONS},
            {"role": "user", "content": transcript},
        ],
        [],   # no tools: we want text, not a tool call
    )
    summary = reply.message.get("content") or "(summary unavailable)"
    summary_message = {"role": "user", "content": "Summary of the conversation so far:\n" + summary}
    return [messages[0], summary_message] + messages[cut:]


def render_transcript(messages: list[dict]) -> str:
    """Turn messages into plain text the summariser can read."""
    lines = []
    for message in messages:
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                lines.append(f"[assistant called] {call['function']['name']}({call['function']['arguments'][:300]})")
        if message.get("content"):
            lines.append(f"[{role}] {message['content'][:1500]}")
    return "\n".join(lines)


def fit_context(messages: list[dict], budget_tokens: int, llm, keep_recent: int = 12) -> list[dict]:
    """Shrink the list until it fits the budget: first trim, then summarise.

    Each step is tried with a generous tail first, then a shorter one, so a
    short-but-heavy conversation can still be shrunk.
    """
    if estimate_tokens(messages) <= budget_tokens:
        return messages
    for keep in (keep_recent, keep_recent // 2, 3):
        messages = trim_old_tool_results(messages, keep)
        if estimate_tokens(messages) <= budget_tokens:
            return messages
    for keep in (keep_recent, 3):
        messages = summarise_older_messages(messages, llm, keep)
        if estimate_tokens(messages) <= budget_tokens:
            return messages
    return messages   # nothing more we can do; the recent tail alone is over budget
