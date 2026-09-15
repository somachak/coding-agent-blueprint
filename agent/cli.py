"""
cli.py - chat with the agent in the terminal.

    python -m agent

Every event the loop reports is printed on its own line, so you can watch
Reason -> Act -> Observe happen in real time.
"""

import sys

from . import create_agent
from .config import load_settings


def print_event(event: dict) -> None:
    kind = event["type"]
    if kind == "model_call":
        print(f"  [reason]  step {event['step']}: asking the model ({event['message_count']} messages so far)")
    elif kind == "tool_call":
        print(f"  [act]     {event['name']}({event['arguments']})")
    elif kind == "tool_result":
        preview = event["result"].replace("\n", " ")[:160]
        print(f"  [observe] {preview}")
    elif kind == "error":
        print(f"  [error]   {event['text']}")


def main() -> None:
    settings = load_settings()
    missing = settings.missing()
    if missing:
        print(f"Missing settings: {', '.join(missing)}. Copy .env.example to .env and fill them in.")
        sys.exit(1)

    agent = create_agent(settings)
    print(f"Coding agent ready. Model: {settings.model}. Workspace: {settings.workspace}")
    print("Type a task. 'reset' clears the conversation, 'exit' quits.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_text == "":
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if user_text.lower() == "reset":
            agent.reset()
            print("Conversation cleared.\n")
            continue

        answer = agent.run(user_text, on_event=print_event)
        print(f"\nAgent: {answer}\n")
        usage = agent.total_usage
        print(f"  (tokens so far: {usage['prompt_tokens']} in, {usage['completion_tokens']} out)\n")


if __name__ == "__main__":
    main()
