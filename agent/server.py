"""
server.py - the agent in a browser window.

    python -m agent.server          ->  http://127.0.0.1:8765

One small HTTP server, standard library only, with four routes:

  GET  /         the single-page UI (web/index.html)
  POST /chat     {"text": "..."}  ->  a stream of loop events (Server-Sent Events)
  GET  /state    the whole conversation the model sees, plus usage and settings
  POST /reset    forget the conversation

How the streaming works: the browser sends one POST. Instead of one JSON
reply at the end, we keep the connection open and write one line per
event as the loop reports it:  "data: {...json...}\\n\\n".  That is the
entire Server-Sent Events format. The page reads the lines as they arrive.

Three rules this file enforces, each in one small function:

  check_origin        a page from another website may not drive this agent
  read_json_object    a request must be a small JSON object, or it is refused
  settings_for_run    settings for one run are chosen BEFORE the run and
                      never changed during it; they become the active
                      settings only if the run succeeds

One agent, one conversation, one run at a time. `run_lock` is what makes
"one at a time" true: /chat and /reset both take it, and a request that
cannot get it is answered 409 and changes nothing.
"""

import dataclasses
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import create_agent, make_llm
from .config import PROJECT_ROOT, load_settings

WEB_DIR = os.path.join(PROJECT_ROOT, "web")
HOST = os.environ.get("AGENT_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGENT_PORT", "8765"))

MAX_BODY_BYTES = 64_000        # a chat request is a sentence or a paragraph, not a file
MAX_TEXT_CHARS = 20_000

# Web pages served from OTHER origins (for example the hosted course site)
# may talk to this server only if their origin is listed here. The same
# list is used twice: to answer the browser's CORS question, and to refuse
# the request itself, because CORS alone does not stop a request running.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("AGENT_ALLOWED_ORIGINS", "https://coding-agent-blueprint.pixelartinc.workers.dev").split(",")
    if origin.strip()
]

# The page may switch provider URL only to OpenRouter or to the URL from
# .env, unless you opt in to anything: AGENT_ALLOW_ANY_PROVIDER=1 for local
# experiments with other OpenAI-compatible servers.
OPENROUTER_URL = "https://openrouter.ai/api/v1"
ALLOW_ANY_PROVIDER = os.environ.get("AGENT_ALLOW_ANY_PROVIDER", "") == "1"

# `serve()` fills these in; the course runs the same server with a fake model.
settings = None          # the ACTIVE settings: what /state reports
agent = None
run_lock = threading.Lock()


class EventStream:
    """Writes loop events to the browser, and keeps going quietly if it left.

    The loop must finish its bookkeeping (every tool call answered) even if
    the tab was closed half way. So a broken connection is remembered in
    `alive`, and later events are simply dropped instead of raising into
    the loop.
    """

    def __init__(self, wfile):
        self.wfile = wfile
        self.alive = True
        self.saw_error = False

    def send(self, event: dict) -> None:
        if event.get("type") == "error":
            self.saw_error = True
        if not self.alive:
            return
        try:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.alive = False


def settings_for_run(active, overrides: dict):
    """Return (settings for this run, problem text or None).

    A copy of the active settings with the page's overrides applied. Rules:
      - a new provider URL must come with its own key (never reuse a key
        across providers);
      - remote provider URLs must be https; plain http only for localhost;
      - only OpenRouter or the URL from .env, unless AGENT_ALLOW_ANY_PROVIDER=1.
    """
    model = str(overrides.get("model", "")).strip()
    base_url = str(overrides.get("base_url", "")).strip().rstrip("/")
    api_key = str(overrides.get("api_key", "")).strip()

    changes = {}
    if model:
        changes["model"] = model
    if base_url and base_url != active.base_url:
        if not api_key:
            return None, "A different provider URL needs its own API key. Enter the URL and the key together."
        problem = check_provider_url(base_url, active.base_url)
        if problem:
            return None, problem
        changes["base_url"] = base_url
    if api_key:
        changes["api_key"] = api_key
    return dataclasses.replace(active, **changes), None


def check_provider_url(url: str, configured_url: str):
    parts = urlsplit(url)
    is_local = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return "The provider URL must start with https://."
    if parts.scheme == "http" and not is_local:
        return "A remote provider URL must use https://, or the key would travel unencrypted."
    if url not in (OPENROUTER_URL, configured_url.rstrip("/")) and not ALLOW_ANY_PROVIDER:
        return ("Only OpenRouter or the provider from .env is allowed from the page. "
                "Start the server with AGENT_ALLOW_ANY_PROVIDER=1 to use another endpoint.")
    return None


class Handler(BaseHTTPRequestHandler):
    # ----- helpers ---------------------------------------------------------

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_cors_headers(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")

    def do_OPTIONS(self) -> None:
        """The browser asks permission before a cross-origin POST. Answer it."""
        self.send_response(204)
        self.send_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def check_origin(self):
        """Return a problem text if this request comes from a page we do not trust.

        No Origin header means a non-browser client (curl, a script): allowed.
        Same site as this server: allowed. Listed origin: allowed. Else 403.
        """
        origin = self.headers.get("Origin")
        if origin is None:
            return None
        host = self.headers.get("Host", "")
        if urlsplit(origin).netloc == host or origin in ALLOWED_ORIGINS:
            return None
        return f"Requests from {origin} are not allowed. Add it to AGENT_ALLOWED_ORIGINS if it is yours."

    def read_json_object(self):
        """Return (object, problem). The body must be a small JSON object."""
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("application/json"):
            return None, "Content-Type must be application/json."
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, "Content-Length is not a number."
        if length > MAX_BODY_BYTES:
            return None, f"The request body is too large (limit {MAX_BODY_BYTES} bytes)."
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, "The request body is not valid JSON."
        if not isinstance(body, dict):
            return None, "The request body must be a JSON object."
        return body, None

    def log_message(self, format, *args):  # noqa: A002 - quieter default logging
        sys.stderr.write("  http: " + (format % args) + "\n")

    # ----- routes ------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.serve_file("index.html", "text/html; charset=utf-8")
        elif self.path == "/state":
            self.send_json(200, {
                "messages": agent.messages,
                "usage": agent.total_usage,
                "model": settings.model,
                "base_url": settings.base_url,
                "workspace": settings.workspace,
                "tools": agent.tools.names(),
                "missing": settings.missing(),
                "busy": run_lock.locked(),
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        problem = self.check_origin()
        if problem:
            self.send_json(403, {"error": problem})
        elif self.path == "/reset":
            self.handle_reset()
        elif self.path == "/chat":
            self.handle_chat()
        else:
            self.send_json(404, {"error": "not found"})

    def serve_file(self, name: str, content_type: str) -> None:
        path = os.path.join(WEB_DIR, name)
        with open(path, "rb") as file:
            body = file.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def handle_reset(self) -> None:
        if not run_lock.acquire(blocking=False):
            self.send_json(409, {"error": "The agent is still working. Wait for it to finish, then reset."})
            return
        try:
            agent.reset()
            self.send_json(200, {"ok": True})
        finally:
            run_lock.release()

    def handle_chat(self) -> None:
        global settings
        body, problem = self.read_json_object()
        if problem:
            self.send_json(400, {"error": problem})
            return
        text = body.get("text")
        if not isinstance(text, str) or text.strip() == "":
            self.send_json(400, {"error": "text must be a non-empty string."})
            return
        if len(text) > MAX_TEXT_CHARS:
            self.send_json(400, {"error": f"text is too long (limit {MAX_TEXT_CHARS} characters)."})
            return
        overrides = body.get("settings") if isinstance(body.get("settings"), dict) else {}

        # Everything below happens with the lock held: the run's settings
        # cannot be changed by another request until this one is over.
        if not run_lock.acquire(blocking=False):
            self.send_json(409, {"error": "The agent is still working on the previous message."})
            return
        try:
            run_settings, problem = settings_for_run(settings, overrides)
            if problem:
                self.send_json(400, {"error": problem})
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_cors_headers()
            self.end_headers()

            stream = EventStream(self.wfile)
            # A fake model (tests, the offline chapters) has no settings and is left alone.
            uses_real_model = hasattr(agent.llm, "settings")
            if uses_real_model:
                agent.llm = make_llm(run_settings)
            try:
                agent.run(text.strip(), on_event=stream.send)
            except Exception as error:  # noqa: BLE001 - report, never die silently
                sys.stderr.write(f"  unexpected error in agent.run: {type(error).__name__}: {error}\n")
                stream.send({"type": "error", "text": f"Unexpected error: {type(error).__name__}: {error}"})
            finally:
                if uses_real_model:
                    agent.llm = make_llm(settings)
            if not stream.saw_error:
                settings = run_settings          # the overrides proved themselves; keep them
            stream.send({"type": "done", "usage": agent.total_usage, "message_count": len(agent.messages),
                         "model": settings.model})
        finally:
            run_lock.release()


def serve(the_agent, the_settings, host: str = HOST, port: int = PORT) -> None:
    """Start the server around any Agent, real or fake, and block until Ctrl+C."""
    global agent, settings
    agent = the_agent
    settings = the_settings
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Agent UI:   http://{host}:{port}")
    print(f"Model:      {settings.model} via {settings.base_url}")
    print(f"Workspace:  {settings.workspace}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    the_settings = load_settings()
    missing = the_settings.missing()
    if missing:
        print(f"Warning: missing settings {', '.join(missing)}. The page will load but the model cannot be called.")
    serve(create_agent(the_settings), the_settings)


if __name__ == "__main__":
    main()
