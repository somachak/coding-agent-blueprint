"""
llm.py - the one place that talks to the model.

Everything the agent learns from the model passes through chat(). It sends
the message list (and the tool menu) to an OpenAI-compatible endpoint and
returns the assistant's reply. No SDK: plain HTTP with the standard
library, so every byte that leaves and arrives is visible.

The request, as JSON:
    {"model": "...", "messages": [...], "tools": [...]}

The reply, as JSON (the parts we use):
    {"choices": [{"message": {"role": "assistant",
                              "content": "text or null",
                              "tool_calls": [...]},      # only when it wants a tool
                  "finish_reason": "stop" | "tool_calls" | "length"}],
     "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}}

Trust nothing in the reply until it has been checked. parse_reply() turns
any malformed reply into an LLMError with a sentence a human can act on,
so the loop only ever sees a well-formed ModelReply or a clean failure.
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
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        raise LLMError(explain_http_error(error)) from error
    except urllib.error.URLError as error:
        raise LLMError(f"Could not reach {settings.base_url}: {error.reason}") from error
    except TimeoutError as error:
        raise LLMError(f"The model did not answer within {settings.timeout_seconds}s.") from error

    return parse_reply(body)


def parse_reply(body: str) -> ModelReply:
    """Check the provider's reply piece by piece and build a ModelReply.

    Every check raises LLMError with a plain sentence. The alternative,
    letting a KeyError escape from deep inside the loop, would crash the
    server and tell you nothing.
    """
    try:
        reply = json.loads(body)
    except json.JSONDecodeError:
        raise LLMError(f"The provider did not return JSON. It said: {body[:300]!r}") from None
    if not isinstance(reply, dict):
        raise LLMError("The provider returned JSON that is not an object.")

    # Some providers return HTTP 200 with an error object inside the body.
    choices = reply.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        detail = reply.get("error", reply)
        raise LLMError(f"The provider returned no choices: {json.dumps(detail)[:500]}")

    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise LLMError(f"The provider's first choice has no message: {json.dumps(choice)[:300]}")

    message = clean_assistant_message(choice["message"])
    finish_reason = choice.get("finish_reason")
    return ModelReply(
        message=message,
        finish_reason=finish_reason if isinstance(finish_reason, str) else "",
        usage=clean_usage(reply.get("usage")),
    )


def clean_assistant_message(raw: dict) -> dict:
    """Keep only the fields we are allowed to send back next turn, checked.

    Providers add extras (reasoning, annotations, refusal ...). Sending those
    back to a *different* provider can cause a 400 error, so we keep the
    three fields the API contract actually defines. A refusal becomes
    visible text; a malformed tool call is an LLMError.
    """
    message = {"role": "assistant"}
    content = raw.get("content")
    if isinstance(content, list):   # a few providers send content as parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if content is not None and not isinstance(content, str):
        content = str(content)

    refusal = raw.get("refusal")
    if isinstance(refusal, str) and refusal.strip() and not content:
        content = "The model refused: " + refusal.strip()

    tool_calls = [clean_tool_call(call) for call in raw.get("tool_calls") or []]

    if content is not None:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
    if content is None and not tool_calls:
        message["content"] = ""
    return message


def clean_tool_call(call) -> dict:
    """A tool call must have an id, a name, and arguments as JSON TEXT."""
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        raise LLMError(f"The provider sent a malformed tool call: {json.dumps(call)[:200]}")
    function = call["function"]
    call_id = call.get("id")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(call_id, str) or call_id == "" or not isinstance(name, str) or name == "":
        raise LLMError(f"The provider sent a tool call without an id or name: {json.dumps(call)[:200]}")
    if isinstance(arguments, dict):       # some providers send the object instead of its text
        arguments = json.dumps(arguments)
    elif arguments is None:
        arguments = "{}"
    elif not isinstance(arguments, str):
        raise LLMError(f"The provider sent tool arguments that are not text: {json.dumps(arguments)[:200]}")
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def clean_usage(raw) -> dict:
    """Token counts as non-negative whole numbers, or zeros if absent or odd."""
    usage = {}
    if isinstance(raw, dict):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw.get(key, 0)
            usage[key] = value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
    else:
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return usage


def explain_http_error(error: urllib.error.HTTPError) -> str:
    """Turn an HTTP status into a sentence that says what to do next."""
    try:
        body = error.read().decode("utf-8", errors="replace")
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
