"""
agent - a bare-bones, model-agnostic coding agent you can read in one sitting.

The package has one job per file:

  config.py     where the settings come from (.env)         -> Settings
  llm.py        the ONE function that talks to the model    -> chat()
  tools.py      what a tool is + the registry that runs it  -> Tool, ToolRegistry
  workspace.py  the five coding tools, locked to one folder -> Workspace, build_tools()
  loop.py       the agent loop: Reason -> Act -> Observe    -> Agent
  context.py    keeping the conversation inside the window  -> fit_context()
  fake_llm.py   a scripted stand-in model for offline runs  -> FakeLLM
  cli.py        chat in the terminal                        -> python -m agent
  server.py     chat in the browser (single window)         -> python -m agent.server

`create_agent()` below wires the real pieces together. Tests and the
course chapters wire them by hand so you can see every connection.
"""

from .config import Settings, load_settings
from .llm import LLMError, chat
from .loop import Agent
from .tools import Tool, ToolRegistry
from .workspace import Workspace, build_tools


def create_agent(settings: Settings | None = None) -> Agent:
    """Build a ready-to-use Agent from the settings in .env.

    This is the only place that knows how the pieces plug together.
    Read it top to bottom and you have the whole architecture:

        settings  ->  llm function  ->  Agent  <-  ToolRegistry  <-  Workspace
    """
    if settings is None:
        settings = load_settings()

    workspace = Workspace(settings.workspace)
    registry = ToolRegistry(build_tools(workspace))
    system_prompt = build_system_prompt(workspace)
    return Agent(
        llm=make_llm(settings),
        tools=registry,
        system_prompt=system_prompt,
        max_steps=settings.max_steps,
        context_window=settings.context_window,
    )


def make_llm(settings: Settings):
    """Bind chat() to one Settings object.

    The loop calls the result with (messages, tool_schemas) and never sees
    the key, the URL or the model name. To run with different settings,
    make a new llm function; the old Settings object is never changed.
    """

    def llm(messages: list[dict], tool_schemas: list[dict]):
        return chat(messages, tool_schemas, settings)

    llm.settings = settings   # lets server.py tell a real model apart from a fake one
    return llm


def build_system_prompt(workspace: Workspace) -> str:
    """The standing instructions the model sees first, every single call.

    If the workspace contains an AGENTS.md file, its text is appended, so a
    project can teach the agent its own rules without touching this code.
    """
    prompt = (
        "You are a careful coding agent working inside one workspace folder.\n"
        "You can only see and change files through your tools. Rules:\n"
        "1. Look before you touch: list_files or read_file before editing.\n"
        "2. Make small, precise edits with edit_file; use write_file only for new files.\n"
        "3. After changing code, run it with run_command to check it works.\n"
        "4. When the job is done, reply with a short plain-language summary of what you did.\n"
        "5. If something fails, read the error, fix the cause, and try again. Do not guess.\n"
        f"Workspace root: {workspace.root}\n"
    )
    instructions = workspace.read_instructions()
    if instructions:
        prompt = prompt + "\nProject instructions (from AGENTS.md):\n" + instructions
    return prompt


__all__ = [
    "Agent",
    "LLMError",
    "Settings",
    "Tool",
    "ToolRegistry",
    "Workspace",
    "build_system_prompt",
    "build_tools",
    "chat",
    "create_agent",
    "load_settings",
    "make_llm",
]
