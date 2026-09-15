"""
workspace.py - the five coding tools, locked to one folder.

A coding agent needs to look around, read, write, edit and run things.
Every one of those actions goes through a Workspace object. The four FILE
tools check every path so the model cannot read or write outside the root
folder. The SHELL tool is different, and honesty matters here:

    run_command starts a real shell as YOU, with your permissions. Its
    working directory is the workspace, but a shell can `cd ..` or read
    any file you can read. The workspace is a convenience for the file
    tools, not a security boundary for the shell.

What this file does about that: the command gets a minimal environment
(no API keys), a clamped timeout, its whole process group is killed on
timeout, and captured output is capped. A real boundary needs an
operating-system sandbox: on the hosted version that is Cloudflare's
container; on your own machine it is out of this blueprint's scope.

Each tool returns TEXT. The model reads that text and decides what to do
next. Errors are also text (see tools.py for why).
"""

import os
import signal
import subprocess
import tempfile

from .tools import Tool

# How much of a tool's output the model gets to see. Huge outputs waste the
# context window and hide the useful part, so we keep the head and the tail.
MAX_OUTPUT_CHARS = 12_000
MAX_LIST_ENTRIES = 300
MAX_WRITE_CHARS = 2_000_000          # one write_file call; bigger is almost certainly a mistake
MAX_INSTRUCTIONS_CHARS = 20_000      # AGENTS.md is a note, not a manual
MAX_COMMAND_SECONDS = 120            # the longest any single command may run
MAX_CAPTURE_BYTES = 200_000          # read at most this much of a command's output into memory
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".wrangler"}


class Workspace:
    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        os.makedirs(self.root, exist_ok=True)

    # ----- safety ---------------------------------------------------------

    def safe_path(self, relative_path: str) -> str:
        """Turn a path the model wrote into a real path INSIDE the root.

        realpath() follows symlinks and removes '..', so a path that looks
        innocent but leads outside the folder is caught here.
        """
        candidate = os.path.realpath(os.path.join(self.root, relative_path))
        is_root = candidate == self.root
        is_inside = candidate.startswith(self.root + os.sep)
        if not (is_root or is_inside):
            raise ValueError(f"{relative_path!r} is outside the workspace and was refused.")
        return candidate

    # ----- the five tools ---------------------------------------------------

    def list_files(self, path: str = ".") -> str:
        """List files and folders under `path` (recursive), one per line."""
        start = self.safe_path(path)
        if not os.path.isdir(start):
            return f"Error: {path!r} is not a folder."
        lines = []
        for folder, subfolders, files in os.walk(start):
            subfolders[:] = sorted(name for name in subfolders if name not in SKIP_DIRS)
            for name in sorted(files):
                full = os.path.join(folder, name)
                shown = os.path.relpath(full, self.root)
                size = os.path.getsize(full)
                lines.append(f"{shown}  ({size} bytes)")
                if len(lines) >= MAX_LIST_ENTRIES:
                    lines.append(f"... stopped after {MAX_LIST_ENTRIES} entries")
                    return "\n".join(lines)
        if not lines:
            return "(the folder is empty)"
        return "\n".join(lines)

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 400) -> str:
        """Return the file's text with line numbers, e.g. '  12| print(x)'.

        Line numbers matter: they let the model quote exact lines back when
        it edits, and let you check its claims against the real file.
        """
        full = self.safe_path(path)
        if not os.path.isfile(full):
            return f"Error: {path!r} does not exist."
        with open(full, encoding="utf-8", errors="replace") as file:
            all_lines = file.read().splitlines()
        first = max(start_line, 1)
        last = min(first + max_lines - 1, len(all_lines))
        numbered = []
        for number in range(first, last + 1):
            numbered.append(f"{number:5d}| {all_lines[number - 1]}")
        header = f"{path} (lines {first}-{last} of {len(all_lines)})\n"
        return truncate_output(header + "\n".join(numbered))

    def write_file(self, path: str, content: str) -> str:
        """Create or completely replace a file."""
        if not isinstance(content, str):
            return "Error: content must be text."
        if len(content) > MAX_WRITE_CHARS:
            return f"Error: content is {len(content)} characters; the limit is {MAX_WRITE_CHARS}. Write the file in parts."
        full = self.safe_path(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as file:
            file.write(content)
        return f"Wrote {len(content)} characters to {path}."

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace ONE exact occurrence of old_text with new_text.

        Exactness is the safety feature. If old_text is missing the model
        is told so; if it appears twice the model must include more context.
        Either way nothing is changed until the match is unambiguous.
        """
        full = self.safe_path(path)
        if not os.path.isfile(full):
            return f"Error: {path!r} does not exist. Use write_file to create it."
        with open(full, encoding="utf-8") as file:
            original = file.read()
        occurrences = original.count(old_text)
        if occurrences == 0:
            return "Error: old_text was not found in the file. Read the file and copy the text exactly."
        if occurrences > 1:
            return f"Error: old_text appears {occurrences} times. Include more surrounding lines so it is unique."
        updated = original.replace(old_text, new_text, 1)
        with open(full, "w", encoding="utf-8") as file:
            file.write(updated)
        return f"Edited {path}: replaced {len(old_text)} characters with {len(new_text)}."

    def run_command(self, command: str, timeout_seconds: int = 60) -> str:
        """Run a shell command with the workspace as its current directory.

        Output and exit code come back as text. Three guards, all visible:
          - the command sees a MINIMAL environment (see minimal_environment),
            so it cannot read the model API key;
          - the timeout is clamped to MAX_COMMAND_SECONDS, and on timeout the
            whole process group is killed, not just the shell;
          - output is captured to temporary files and only the first
            MAX_CAPTURE_BYTES are read back, so a runaway command cannot
            fill memory.
        None of this stops the shell from leaving the folder. See the note
        at the top of this file.
        """
        if not isinstance(command, str) or command.strip() == "":
            return "Error: command must be a non-empty string."
        timeout_seconds = clamp_timeout(timeout_seconds)

        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=self.root,
                stdout=stdout_file,
                stderr=stderr_file,
                env=minimal_environment(self.root),
                start_new_session=True,   # the shell becomes a group leader, so we can kill the whole group
            )
            timed_out = False
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            stdout = read_captured(stdout_file)
            stderr = read_captured(stderr_file)

        output = stdout
        if stderr:
            output = output + ("\n[stderr]\n" if output else "[stderr]\n") + stderr
        if output.strip() == "":
            output = "(no output)"
        if timed_out:
            return truncate_output(
                f"Error: the command did not finish within {timeout_seconds} seconds and was killed.\n"
                f"Output before the kill:\n{output}"
            )
        return truncate_output(f"exit code {process.returncode}\n{output}")

    # ----- project memory ----------------------------------------------------

    def read_instructions(self) -> str:
        """Return the text of AGENTS.md in the workspace, or '' if there is none.

        Goes through safe_path like every other read, so a symlink named
        AGENTS.md that points outside the workspace is ignored, and the
        text is capped so a huge file cannot swamp the system prompt.
        """
        try:
            path = self.safe_path("AGENTS.md")
        except ValueError:
            return ""
        if not os.path.isfile(path):
            return ""
        with open(path, encoding="utf-8", errors="replace") as file:
            text = file.read(MAX_INSTRUCTIONS_CHARS + 1)
        if len(text) > MAX_INSTRUCTIONS_CHARS:
            text = text[:MAX_INSTRUCTIONS_CHARS] + "\n[AGENTS.md was cut here: keep it short]"
        return text.strip()


def clamp_timeout(value) -> int:
    """Turn whatever the model sent into a whole number of seconds within limits."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        seconds = 60
    return max(1, min(seconds, MAX_COMMAND_SECONDS))


def minimal_environment(home: str) -> dict:
    """The only environment variables a command gets.

    Notably absent: LLM_API_KEY and everything else from .env. A command the
    model wrote must never be able to print the key. HOME points at the
    workspace so tools that write config files stay inside it by default.
    """
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": home,
        "LANG": "C.UTF-8",
        "TERM": "dumb",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def read_captured(file) -> str:
    """Read at most MAX_CAPTURE_BYTES from a temporary output file."""
    file.seek(0)
    data = file.read(MAX_CAPTURE_BYTES + 1)
    text = data[:MAX_CAPTURE_BYTES].decode("utf-8", errors="replace")
    if len(data) > MAX_CAPTURE_BYTES:
        text += f"\n... [output beyond {MAX_CAPTURE_BYTES} bytes was not captured] ..."
    return text


def truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the start and the end of a long text, and say how much was cut."""
    if len(text) <= limit:
        return text
    keep = limit // 2
    cut = len(text) - limit
    return text[:keep] + f"\n... [{cut} characters cut out of the middle] ...\n" + text[-keep:]


def build_tools(workspace: Workspace) -> list[Tool]:
    """Describe the five workspace methods so the model can call them.

    This is the menu. The description is the ONLY thing the model reads to
    decide when to use a tool, so it is written for the model, not for you.
    """
    return [
        Tool(
            name="list_files",
            description="List every file under a folder in the workspace, with sizes. Start here.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Folder to list, relative to the workspace. Default '.'"},
                },
                "required": [],
            },
            function=workspace.list_files,
        ),
        Tool(
            name="read_file",
            description="Read a text file with line numbers. Use start_line and max_lines for long files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "start_line": {"type": "integer", "description": "First line to show (1-based). Default 1."},
                    "max_lines": {"type": "integer", "description": "How many lines to show. Default 400."},
                },
                "required": ["path"],
            },
            function=workspace.read_file,
        ),
        Tool(
            name="write_file",
            description="Create a new file or completely replace an existing one with the given content.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "content": {"type": "string", "description": "The full text the file should contain."},
                },
                "required": ["path", "content"],
            },
            function=workspace.write_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace one exact occurrence of old_text with new_text in a file. "
                "old_text must match the file exactly (whitespace included) and appear only once."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "old_text": {"type": "string", "description": "The exact text to find."},
                    "new_text": {"type": "string", "description": "The text to put in its place."},
                },
                "required": ["path", "old_text", "new_text"],
            },
            function=workspace.edit_file,
        ),
        Tool(
            name="run_command",
            description=(
                "Run a shell command with the workspace as the current directory (for example "
                "'python3 app.py' or 'ls -la') and return its output and exit code. Commands are "
                f"killed after timeout_seconds (maximum {MAX_COMMAND_SECONDS})."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command line to run."},
                    "timeout_seconds": {"type": "integer", "description": "Kill it after this many seconds. Default 60."},
                },
                "required": ["command"],
            },
            function=workspace.run_command,
        ),
    ]
