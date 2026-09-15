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
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import create_agent
from .config import PROJECT_ROOT, load_settings

WEB_DIR = os.path.join(PROJECT_ROOT, "web")
HOST = os.environ.get("AGENT_HOST", "127.0.0.1")
PORT = int(os.environ.get("AGENT_PORT", "8765"))

# Web pages served from OTHER origins (for example the hosted course site)
# may talk to this server only if their origin is listed here. Anything
# else is refused by the browser, which is what keeps a random website from
# driving the agent on your machine.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("AGENT_ALLOWED_ORIGINS", "https://coding-agent-blueprint.pixelartinc.workers.dev").split(",")
    if origin.strip()
]

# One agent, one conversation, one run at a time. Fine for a local tool.
# A multi-user deployment would keep one agent per session id instead.
# `serve()` fills these in; the course runs the same server with a fake model.
settings = None
agent = None
run_lock = threading.Lock()


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

    def apply_setting_overrides(self, overrides: dict) -> None:
        """Let the page choose the key, endpoint and model for this conversation.

        The values live only in memory, in this process. Nothing is written
        to disk. The llm function reads `settings` on every call, so the
        change takes effect on the very next model call.
        """
        for name in ("api_key", "base_url", "model"):
            value = str(overrides.get(name, "")).strip()
            if value:
                setattr(settings, name, value)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

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
            })
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/reset":
            agent.reset()
            self.send_json(200, {"ok": True})
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

    def handle_chat(self) -> None:
        body = self.read_json_body()
        text = str(body.get("text", "")).strip()
        if text == "":
            self.send_json(400, {"error": "text is required"})
            return
        if isinstance(body.get("settings"), dict):
            self.apply_setting_overrides(body["settings"])
        if not run_lock.acquire(blocking=False):
            self.send_json(409, {"error": "the agent is still working on the previous message"})
            return

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_cors_headers()
            self.end_headers()

            def send_event(event: dict) -> None:
                line = "data: " + json.dumps(event) + "\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            try:
                agent.run(text, on_event=send_event)
            except (BrokenPipeError, ConnectionResetError):
                pass   # the browser tab was closed mid-run; nothing to send to
            else:
                send_event({"type": "done", "usage": agent.total_usage, "message_count": len(agent.messages)})
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
