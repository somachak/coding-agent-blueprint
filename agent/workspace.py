"""
workspace.py - the five coding tools, locked to one folder.

A coding agent needs to look around, read, write, edit and run things.
Every one of those actions goes through a Workspace object, and every
path is checked so the agent cannot wander outside its root folder.

Each tool returns TEXT. The model reads that text and decides what to do
next. Errors are also text (see tools.py for why).
"""

import os
import subprocess

from .tools import Tool

# How much of a tool's output the model gets to see. Huge outputs waste the
# context window and hide the useful part, so we keep the head and the tail.
MAX_OUTPUT_CHARS = 12_000
MAX_LIST_ENTRIES = 300
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
        """Run a shell command inside the workspace and return its output.

        stdout and stderr are both captured, the exit code is reported, and a
        timeout stops runaway commands. The command runs with the same rights
        as you, so the folder lock is a convenience, not a security boundary.
        """
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return f"Error: the command did not finish within {timeout_seconds} seconds and was killed."
        output = completed.stdout
        if completed.stderr:
            output = output + ("\n[stderr]\n" if output else "[stderr]\n") + completed.stderr
        if output.strip() == "":
            output = "(no output)"
        return truncate_output(f"exit code {completed.returncode}\n{output}")

    # ----- project memory ----------------------------------------------------

    def read_instructions(self) -> str:
        """Return the text of AGENTS.md in the workspace, or '' if there is none."""
        path = os.path.join(self.root, "AGENTS.md")
        if not os.path.isfile(path):
            return ""
        with open(path, encoding="utf-8") as file:
            return file.read().strip()


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
                "Run a shell command inside the workspace (for example 'python3 app.py' or 'ls -la') "
                "and return its output and exit code."
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
