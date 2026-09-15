# Coding agent blueprint

A bare-bones, model-agnostic coding agent in plain Python (standard library only), built as a ten-chapter course that ends in a package you can drop into your own app. One chat window shows every step the agent takes.

**Course site:** `site/index.html` (built from `docs/course.html`) · **Chat window:** `python3 -m agent.server`

## What is in here

| Folder | What |
|---|---|
| `agent/` | The blueprint. One job per file: `llm.py` talks to the model, `tools.py` is the registry, `workspace.py` the five coding tools, `loop.py` the loop, `context.py` compaction, `server.py` + `web/index.html` the window. |
| `course/` | Ten runnable chapters. 1 to 5 grow one script from a single HTTP call to a coding agent; 6 to 10 use the package (offline, no key needed). |
| `tests/` | 71 tests on a scripted fake model and a real local socket. `python3 -m unittest -v` |
| `docs/` | The course narrative and the site builder that embeds the real source files. |
| `deploy/cloudflare/` | A Worker that hosts the site and runs the same Python inside a Cloudflare Sandbox. |
| `workspace/` | The only folder the agent may touch. |

## Run it

```bash
cp .env.example .env          # paste an OpenRouter key into LLM_API_KEY
python3 -m agent              # terminal chat
python3 -m agent.server       # http://127.0.0.1:8765
```

Any OpenAI-compatible endpoint works: change `LLM_BASE_URL` and `LLM_MODEL`. The model must support tool calling. The default is `openai/gpt-5-mini`.

## Check it

```bash
python3 -m unittest -v                       # all tests, verbose
python3 -m unittest discover -s tests -t .   # the same, spelled out
python3 tests/test_agent.py                  # the test file on its own
cd deploy/cloudflare && npm run typecheck && npm test   # the Worker
```

## What is, and is not, a safety boundary

- The four file tools (`list_files`, `read_file`, `write_file`, `edit_file`) check every path and refuse anything outside `workspace/`, symlinks included.
- `run_command` is a real shell running as **you**, with your permissions. Its working directory is the workspace, but a shell can leave a directory. What the blueprint does: a minimal environment (no API key), a clamped timeout with the whole process group killed, and capped output. A real boundary needs an operating-system sandbox: on the hosted version that is Cloudflare's container; on your own machine it is outside this blueprint's scope.
- The web page keeps the API key in the page's memory only (never localStorage); the local server refuses requests from unlisted origins and validates every request body; settings for one run are fixed before it starts and become active only if it succeeds.

## The course, one mechanism per chapter

| # | Chapter | Adds |
|---|---|---|
| 1 | One call | HTTP request, the message dictionary, JSON in and out |
| 2 | The list that grows | `while True`, append everything, memory lives in the list |
| 3 | The first tool | The tool schema, `tool_calls`, the tool message, `tool_call_id` |
| 4 | The loop | Stop on no tool calls; a step brake |
| 5 | Registry + coding tools | name → function table, errors as text, safe paths, truncation, timeouts |
| 6 | Events → the package | `on_event`, the `Agent` class, the fake model |
| 7 | The web window | A stdlib server streaming events (SSE) to one page |
| 8 | Context | System prompt + `AGENTS.md`; trim then summarise when the list grows |
| 9 | Troubleshooting lab | Twelve failures reproduced offline, with the fix for each |
| 10 | Your own tools | Swap the tools, keep the loop: the transplant recipe |

Each chapter file starts with "what breaks if you remove it". That list is the troubleshooting course.

## Deploy

```bash
cd deploy/cloudflare && npm install
python3 ../../docs/build_site.py && python3 bundle.py
npm run deploy:site        # free plan: the course + chat page; agent runs on your machine
npm run deploy:sandbox     # Workers Paid: the agent runs in a per-session Linux sandbox
```

## Credits

Inspired by the open-source course "Building a Coding Agent From Scratch" (Paul Iusztin, Decoding AI), Pi (Mario Zechner) and Thorsten Ball's "How to Build an Agent". This version removes every framework so each line answers to you.
