"""
llm.py - the one place that talks to the model.

Everything the agent learns from the model passes through chat(). It sends
the message list (and the tool menu) to an OpenAI-compatible endpoint and
returns the assistant's reply exactly as it arrived. No SDK: plain HTTP with
the standard library, so every byte that leaves and arrives is visible.

The request, as JSON:
    {"model": "...", "messages": [...], "tools": [...]}

The reply, as JSON (the parts we use):
    {"choices": [{"message": {"role": "assistant",
                              "content": "text or null",
                              "tool_calls": [...]},      # only when it wants a tool
                  "finish_reason": "stop" | "tool_calls"}],
     "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}}
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import Settings


class LLMError(Exception):
    """The model call failed. The message is written for a human to act on."""


@dataclass
class ModelReply:
    """What chat() hands back: the message to append, plus bookkeeping."""

    message: dict
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)


def chat(messages: list[dict], tools: list[dict], settings: Settings) -> ModelReply:
    """One round trip: messages in, the assistant's next message out."""
    if settings.api_key == "":
        raise LLMError("No API key. Copy .env.example to .env and set LLM_API_KEY.")

    payload = {"model": settings.model, "messages": messages}
    if tools:
        payload["tools"] = tools

    request = urllib.request.Request(
        url=settings.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + settings.api_key,
            "Content-Type": "application/json",
            # OpenRouter shows these two on its dashboard; other providers ignore them.
            "HTTP-Referer": "https://github.com/somachak/coding-agent-blueprint",
            "X-Title": "coding-agent-blueprint",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            reply = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise LLMError(explain_http_error(error)) from error
    except urllib.error.URLError as error:
        raise LLMError(f"Could not reach {settings.base_url}: {error.reason}") from error
    except TimeoutError as error:
        raise LLMError(f"The model did not answer within {settings.timeout_seconds}s.") from error

    # Some providers return HTTP 200 with an error object inside the body.
    if "choices" not in reply or len(reply["choices"]) == 0:
        detail = reply.get("error", reply)
        raise LLMError(f"The provider returned no choices: {json.dumps(detail)[:500]}")

    choice = reply["choices"][0]
    message = clean_assistant_message(choice["message"])
    return ModelReply(
        message=message,
        finish_reason=choice.get("finish_reason", ""),
        usage=reply.get("usage", {}),
    )


def clean_assistant_message(raw: dict) -> dict:
    """Keep only the fields we are allowed to send back next turn.

    Providers add extras (reasoning, annotations, refusal ...). Sending those
    back to a *different* provider can cause a 400 error, so we keep the
    three fields the API contract actually defines.
    """
    message = {"role": "assistant"}
    content = raw.get("content")
    tool_calls = raw.get("tool_calls")
    if content is not None:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
    if content is None and not tool_calls:
        message["content"] = ""
    return message


def explain_http_error(error: urllib.error.HTTPError) -> str:
    """Turn an HTTP status into a sentence that says what to do next."""
    try:
        body = error.read().decode("utf-8")
    except Exception:
        body = ""
    hints = {
        400: "The request was rejected. Usually a bad model id, or a broken message order "
        "(a tool result without its tool call, or a tool_call_id that does not match).",
        401: "The API key was refused. Check LLM_API_KEY in .env.",
        402: "Out of credits on this provider account.",
        403: "This key is not allowed to use this model.",
        404: "Model or endpoint not found. Check LLM_MODEL and LLM_BASE_URL.",
        408: "The provider timed out. Try again.",
        429: "Rate limited or the free tier is exhausted. Wait, or pick another model.",
        500: "The provider had an internal error. Try again.",
        502: "The provider behind the router failed. Try again or pick another model.",
        503: "The model is temporarily unavailable. Try again shortly.",
    }
    hint = hints.get(error.code, "Unexpected HTTP error.")
    return f"HTTP {error.code}: {hint}\nServer said: {body[:600]}"
