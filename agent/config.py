"""
config.py - where the settings come from.

The agent needs a handful of facts before it can do anything:

  LLM_API_KEY       the secret that proves you may use the model
  LLM_BASE_URL      which server to send requests to (OpenRouter by default)
  LLM_MODEL         which model to ask for
  AGENT_WORKSPACE   the only folder the agent may touch
  AGENT_MAX_STEPS   how many model calls one message may trigger (a brake)
  AGENT_CONTEXT_WINDOW  roughly how many tokens the model can read at once

They live in a .env file so they never sit inside the code or on GitHub.
This file reads .env by hand, in a few lines, so you can see there is no
magic: a .env file is just NAME=value lines.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# The folder that contains this package's parent, i.e. the project root.
# __file__ is this file; .parent is agent/; .parent again is the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Read NAME=value lines from a file and put them into os.environ.

    Values already present in the environment are kept, so you can override
    the file from the terminal for a one-off run:  LLM_MODEL=x python -m agent
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        is_blank = line == ""
        is_comment = line.startswith("#")
        has_equals = "=" in line
        if is_blank or is_comment or not has_equals:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if name not in os.environ:
            os.environ[name] = value


@dataclass
class Settings:
    """Everything the rest of the package needs, in one plain object."""

    api_key: str
    base_url: str
    model: str
    workspace: str
    max_steps: int
    context_window: int = 120_000
    timeout_seconds: int = 120

    def missing(self) -> list[str]:
        """Names of settings that are still empty. Empty list means ready."""
        problems = []
        if self.api_key == "" or self.api_key.startswith("paste-your"):
            problems.append("LLM_API_KEY")
        if self.model == "":
            problems.append("LLM_MODEL")
        return problems


def load_settings(env_file: Path | None = None) -> Settings:
    """Load .env (if present) and build a Settings object from the environment."""
    if env_file is None:
        env_file = PROJECT_ROOT / ".env"
    load_dotenv(env_file)

    # OPENROUTER_API_KEY is accepted as a convenience; LLM_API_KEY wins.
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.environ.get("LLM_MODEL", "")

    workspace = os.environ.get("AGENT_WORKSPACE", "workspace")
    if not os.path.isabs(workspace):
        workspace = str(PROJECT_ROOT / workspace)

    max_steps = int(os.environ.get("AGENT_MAX_STEPS", "25"))
    context_window = int(os.environ.get("AGENT_CONTEXT_WINDOW", "120000"))

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        workspace=workspace,
        max_steps=max_steps,
        context_window=context_window,
    )
