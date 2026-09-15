"""
Tests for the blueprint. Standard library unittest, no API key, no internet.

    python3 -m unittest -v                      # from the project folder
    python3 -m unittest discover -s tests -t .  # the same, spelled out
    python3 tests/test_agent.py                 # run this file directly

Each test is written so that its name tells you what would break.
"""

import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import server as server_module  # noqa: E402
from agent.config import Settings, load_dotenv  # noqa: E402
from agent.context import find_cut_index, summarise_older_messages, trim_old_tool_results  # noqa: E402
from agent.debug import check_message_order  # noqa: E402
from agent.fake_llm import FakeLLM  # noqa: E402
from agent.llm import LLMError, ModelReply, clean_assistant_message, parse_reply  # noqa: E402
from agent.loop import MAX_TOOL_CALLS_PER_REPLY, Agent  # noqa: E402
from agent.tools import Tool, ToolRegistry  # noqa: E402
from agent.workspace import (  # noqa: E402
    MAX_CAPTURE_BYTES,
    MAX_COMMAND_SECONDS,
    MAX_INSTRUCTIONS_CHARS,
    MAX_WRITE_CHARS,
    Workspace,
    build_tools,
    clamp_timeout,
    truncate_output,
)


# ----- helpers shared by the tests ------------------------------------------------

def temp_dir(test: unittest.TestCase, prefix: str = "agent-test-") -> str:
    """A temporary folder that is deleted when the test ends."""
    path = tempfile.mkdtemp(prefix=prefix)
    test.addCleanup(shutil.rmtree, path, True)
    return path


def make_workspace(test: unittest.TestCase) -> Workspace:
    return Workspace(temp_dir(test))


def make_settings(**changes) -> Settings:
    base = dict(api_key="test-key-not-real", base_url="https://openrouter.ai/api/v1", model="openai/gpt-5-mini",
                workspace="unused", max_steps=8)
    base.update(changes)
    return Settings(**base)


class FailingLLM:
    """A model that always fails the way a bad key or bad model name does."""

    def __init__(self, message="HTTP 404: Model or endpoint not found."):
        self.message = message

    def __call__(self, messages, tools):
        raise LLMError(self.message)


class BlockingLLM:
    """Wraps another model and waits for permission before every answer."""

    def __init__(self, inner):
        self.inner = inner
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, messages, tools):
        self.started.set()
        self.release.wait(timeout=10)
        return self.inner(messages, tools)


# ================================================================================
class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry([
            Tool("shout", "upper-case text", {"type": "object", "properties": {"text": {"type": "string"}}}, lambda text: text.upper()),
            Tool("explode", "always fails", {"type": "object", "properties": {}}, lambda: 1 / 0),
        ])

    def test_schema_shape_matches_openai_contract(self):
        schema = self.registry.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "shout")
        self.assertIn("parameters", schema["function"])

    def test_run_calls_the_function_with_parsed_arguments(self):
        self.assertEqual(self.registry.run("shout", '{"text": "hi"}'), "HI")

    def test_unknown_tool_becomes_text_not_exception(self):
        self.assertTrue(self.registry.run("nope", "{}").startswith("Error: there is no tool called 'nope'"))

    def test_bad_json_becomes_text(self):
        self.assertIn("not valid JSON", self.registry.run("shout", "{oops"))

    def test_wrong_argument_name_becomes_text(self):
        self.assertIn("wrong arguments", self.registry.run("shout", '{"txt": "hi"}'))

    def test_exception_inside_tool_becomes_text(self):
        self.assertIn("ZeroDivisionError", self.registry.run("explode", "{}"))

    def test_duplicate_name_is_refused(self):
        with self.assertRaises(ValueError):
            self.registry.add(Tool("shout", "", {}, lambda: ""))


# ================================================================================
class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_workspace(self)

    def test_paths_outside_root_are_refused(self):
        with self.assertRaises(ValueError):
            self.ws.safe_path("../../etc/passwd")

    def test_symlink_pointing_outside_is_refused(self):
        outside_dir = temp_dir(self, "outside-")
        outside = os.path.join(outside_dir, "secret.txt")
        with open(outside, "w") as file:
            file.write("secret")
        os.symlink(outside, os.path.join(self.ws.root, "sneaky.txt"))
        # safe_path raises on purpose (a refusal must never be silent);
        # the registry is what turns that into text for the model.
        registry = ToolRegistry(build_tools(self.ws))
        text = registry.run("read_file", '{"path": "sneaky.txt"}')
        self.assertIn("outside the workspace", text)
        self.assertNotIn("secret", text)

    def test_write_then_read_shows_line_numbers(self):
        self.ws.write_file("a.py", "x = 1\ny = 2\n")
        text = self.ws.read_file("a.py")
        self.assertIn("    1| x = 1", text)
        self.assertIn("    2| y = 2", text)

    def test_read_missing_file_is_an_error_message(self):
        self.assertTrue(self.ws.read_file("ghost.txt").startswith("Error"))

    def test_edit_requires_exactly_one_match(self):
        self.ws.write_file("a.txt", "one two two")
        self.assertIn("not found", self.ws.edit_file("a.txt", "three", "x"))
        self.assertIn("appears 2 times", self.ws.edit_file("a.txt", "two", "x"))
        self.assertIn("Edited", self.ws.edit_file("a.txt", "one", "1"))
        with open(os.path.join(self.ws.root, "a.txt")) as file:
            self.assertEqual(file.read(), "1 two two")

    def test_oversized_write_is_refused_as_text(self):
        text = self.ws.write_file("big.txt", "x" * (MAX_WRITE_CHARS + 1))
        self.assertTrue(text.startswith("Error"))
        self.assertFalse(os.path.exists(os.path.join(self.ws.root, "big.txt")))

    def test_run_command_reports_exit_code_and_stderr(self):
        text = self.ws.run_command("echo out; echo err 1>&2; exit 3")
        self.assertIn("exit code 3", text)
        self.assertIn("out", text)
        self.assertIn("[stderr]", text)

    def test_list_files_skips_noise_folders(self):
        self.ws.write_file("keep.txt", "k")
        self.ws.write_file("__pycache__/junk.pyc", "j")
        listing = self.ws.list_files()
        self.assertIn("keep.txt", listing)
        self.assertNotIn("junk.pyc", listing)

    def test_truncate_keeps_head_and_tail(self):
        text = truncate_output("a" * 100 + "b" * 100, limit=50)
        self.assertTrue(text.startswith("a" * 25))
        self.assertTrue(text.endswith("b" * 25))
        self.assertIn("150 characters cut", text)

    def test_build_tools_names(self):
        names = [tool.name for tool in build_tools(self.ws)]
        self.assertEqual(names, ["list_files", "read_file", "write_file", "edit_file", "run_command"])


# ================================================================================
class CommandBoundaryTests(unittest.TestCase):
    """run_command is a real shell. These pin down the guards around it."""

    def setUp(self):
        self.ws = make_workspace(self)

    def test_command_cannot_see_the_api_key(self):
        os.environ["LLM_API_KEY"] = "sk-test-secret-value"
        self.addCleanup(os.environ.pop, "LLM_API_KEY", None)
        text = self.ws.run_command("env")
        self.assertNotIn("sk-test-secret-value", text)
        self.assertNotIn("LLM_API_KEY", text)
        self.assertIn("PATH=", text)

    def test_home_points_inside_the_workspace(self):
        self.assertIn(self.ws.root, self.ws.run_command("echo $HOME"))

    def test_timeout_is_clamped_to_the_documented_maximum(self):
        self.assertEqual(clamp_timeout(999999), MAX_COMMAND_SECONDS)
        self.assertEqual(clamp_timeout(-5), 1)
        self.assertEqual(clamp_timeout("nonsense"), 60)

    def test_timeout_kills_the_whole_process_group(self):
        started = time.monotonic()
        text = self.ws.run_command("sleep 30 & sleep 30; echo never", timeout_seconds=1)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIn("did not finish within 1 seconds", text)
        self.assertNotIn("never", text)

    def test_captured_output_is_capped(self):
        text = self.ws.run_command(f"head -c {MAX_CAPTURE_BYTES * 3} /dev/zero | tr '\\0' 'x'")
        self.assertIn("was not captured", text)
        self.assertLess(len(text), MAX_CAPTURE_BYTES)


# ================================================================================
class InstructionsTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_workspace(self)

    def test_agents_md_is_read(self):
        self.ws.write_file("AGENTS.md", "Always add tests.")
        self.assertEqual(self.ws.read_instructions(), "Always add tests.")

    def test_agents_md_symlink_outside_is_ignored(self):
        outside_dir = temp_dir(self, "outside-")
        outside = os.path.join(outside_dir, "AGENTS.md")
        with open(outside, "w") as file:
            file.write("SECRET INSTRUCTIONS")
        os.symlink(outside, os.path.join(self.ws.root, "AGENTS.md"))
        self.assertEqual(self.ws.read_instructions(), "")

    def test_agents_md_is_capped(self):
        self.ws.write_file("AGENTS.md", "x" * (MAX_INSTRUCTIONS_CHARS * 2))
        text = self.ws.read_instructions()
        self.assertLess(len(text), MAX_INSTRUCTIONS_CHARS + 100)
        self.assertIn("cut here", text)


# ================================================================================
class LoopTests(unittest.TestCase):
    def make_agent(self, script, max_steps=5):
        ws = make_workspace(self)
        registry = ToolRegistry(build_tools(ws))
        return Agent(llm=FakeLLM(script), tools=registry, system_prompt="sys", max_steps=max_steps), ws

    def test_plain_answer_ends_after_one_step(self):
        agent, _ = self.make_agent(["hello"])
        self.assertEqual(agent.run("hi"), "hello")
        self.assertEqual([m["role"] for m in agent.messages], ["system", "user", "assistant"])

    def test_tool_call_then_answer(self):
        agent, ws = self.make_agent([
            [("write_file", {"path": "x.txt", "content": "hi"})],
            "done",
        ])
        events = []
        answer = agent.run("write x", on_event=events.append)
        self.assertEqual(answer, "done")
        self.assertTrue(os.path.exists(os.path.join(ws.root, "x.txt")))
        self.assertEqual([m["role"] for m in agent.messages], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual([e["type"] for e in events],
                         ["user", "model_call", "model_reply", "tool_call", "tool_result", "model_call", "model_reply", "answer"])

    def test_every_tool_message_answers_a_real_tool_call(self):
        agent, _ = self.make_agent([
            [("list_files", {}), ("read_file", {"path": "nothing.txt"})],
            "ok",
        ])
        agent.run("go")
        assistant = agent.messages[2]
        tool_messages = agent.messages[3:5]
        ids_requested = [call["id"] for call in assistant["tool_calls"]]
        ids_answered = [m["tool_call_id"] for m in tool_messages]
        self.assertEqual(ids_requested, ids_answered)
        self.assertEqual(check_message_order(agent.messages), [])

    def test_step_budget_stops_a_runaway_loop(self):
        endless = [[("list_files", {})]] * 10
        agent, _ = self.make_agent(endless, max_steps=3)
        answer = agent.run("loop forever")
        self.assertIn("Stopped after 3 steps", answer)
        self.assertEqual(check_message_order(agent.messages), [])

    def test_reset_keeps_only_system_prompt(self):
        agent, _ = self.make_agent(["a", "b"])
        agent.run("1")
        agent.reset()
        self.assertEqual(len(agent.messages), 1)

    def test_cut_off_reply_does_not_run_its_tool_calls(self):
        ws = make_workspace(self)

        def cut_off_llm(messages, tools):
            if len(messages) == 2:
                return ModelReply(
                    message={"role": "assistant", "content": None, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "a.txt", "content": "hal'}}]},
                    finish_reason="length")
            return ModelReply(message={"role": "assistant", "content": "ok"}, finish_reason="stop")

        agent = Agent(llm=cut_off_llm, tools=ToolRegistry(build_tools(ws)), system_prompt="sys")
        agent.run("go")
        self.assertFalse(os.path.exists(os.path.join(ws.root, "a.txt")))
        self.assertIn("cut off", agent.messages[3]["content"])
        self.assertEqual(check_message_order(agent.messages), [])

    def test_too_many_tool_calls_in_one_reply_are_refused_but_answered(self):
        calls = [("list_files", {})] * (MAX_TOOL_CALLS_PER_REPLY + 5)
        agent, _ = self.make_agent([calls, "ok"])
        agent.run("go")
        tool_messages = [m for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), MAX_TOOL_CALLS_PER_REPLY + 5)
        self.assertTrue(tool_messages[-1]["content"].startswith("Error: more than"))
        self.assertEqual(check_message_order(agent.messages), [])


# ================================================================================
class LoopFailureTests(unittest.TestCase):
    """A failed turn must leave no trace in the message list."""

    def test_model_error_rolls_the_turn_back(self):
        agent = Agent(llm=FailingLLM(), tools=ToolRegistry(), system_prompt="sys")
        before = list(agent.messages)
        events = []
        answer = agent.run("do the thing", on_event=events.append)
        self.assertTrue(answer.startswith("Error:"))
        self.assertEqual(agent.messages, before)
        self.assertEqual(check_message_order(agent.messages), [])
        self.assertIn("rolled back", events[-1]["text"])

    def test_next_task_does_not_continue_the_failed_task(self):
        calls = {"n": 0}
        fake = FakeLLM(["second answer"])

        def flaky(messages, tools):
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMError("HTTP 401: The API key was refused.")
            return fake(messages, tools)

        agent = Agent(llm=flaky, tools=ToolRegistry(), system_prompt="sys")
        agent.run("FAILED TASK")
        agent.run("second task")
        seen_by_model = json.dumps(fake.calls[0])
        self.assertNotIn("FAILED TASK", seen_by_model)
        self.assertEqual([m["role"] for m in agent.messages], ["system", "user", "assistant"])
        self.assertEqual(agent.messages[1]["content"], "second task")

    def test_error_after_tool_calls_removes_the_whole_turn(self):
        ws = make_workspace(self)
        fake = FakeLLM([[("write_file", {"path": "made.txt", "content": "x"})]])

        def fail_second(messages, tools):
            if len(messages) > 2:
                raise LLMError("HTTP 500: The provider had an internal error.")
            return fake(messages, tools)

        agent = Agent(llm=fail_second, tools=ToolRegistry(build_tools(ws)), system_prompt="sys")
        agent.run("make a file")
        self.assertEqual(len(agent.messages), 1)                      # the turn is gone
        self.assertTrue(os.path.exists(os.path.join(ws.root, "made.txt")))   # the disk is not
        self.assertEqual(check_message_order(agent.messages), [])

    def test_compaction_failure_becomes_an_error_event_and_the_turn_continues(self):
        ws = make_workspace(self)
        ws.write_file("big.txt", "lorem " * 800)
        script = [[("read_file", {"path": "big.txt"})]] * 4 + ["finished"]
        fake = FakeLLM(script)

        def llm(messages, tools):
            if not tools:                       # the summariser asks with no tools
                raise LLMError("HTTP 429: Rate limited.")
            return fake(messages, tools)

        agent = Agent(llm=llm, tools=ToolRegistry(build_tools(ws)), system_prompt="sys", max_steps=10, context_window=1500)
        events = []
        answer = agent.run("read it a lot", on_event=events.append)
        self.assertEqual(answer, "finished")
        self.assertTrue(any(e["type"] == "error" and "Compaction failed" in e["text"] for e in events))
        self.assertEqual(check_message_order(agent.messages), [])


# ================================================================================
class ReplyParsingTests(unittest.TestCase):
    def good(self, **changes):
        reply = {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        reply.update(changes)
        return json.dumps(reply)

    def test_good_reply_parses(self):
        parsed = parse_reply(self.good())
        self.assertEqual(parsed.message["content"], "hi")
        self.assertEqual(parsed.finish_reason, "stop")
        self.assertEqual(parsed.usage["prompt_tokens"], 3)

    def test_not_json_is_an_llm_error(self):
        with self.assertRaises(LLMError):
            parse_reply("<html>gateway timeout</html>")

    def test_missing_choices_is_an_llm_error(self):
        with self.assertRaises(LLMError):
            parse_reply(json.dumps({"error": {"message": "no such model"}}))
        with self.assertRaises(LLMError):
            parse_reply(self.good(choices=[]))
        with self.assertRaises(LLMError):
            parse_reply(self.good(choices="nope"))

    def test_missing_assistant_message_is_an_llm_error(self):
        with self.assertRaises(LLMError):
            parse_reply(self.good(choices=[{"finish_reason": "stop"}]))

    def test_malformed_tool_call_is_an_llm_error(self):
        bad = {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"arguments": "{}"}}]}
        with self.assertRaises(LLMError):
            parse_reply(self.good(choices=[{"message": bad}]))

    def test_dict_arguments_are_turned_into_text(self):
        raw = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": {"path": "a"}}}]}
        cleaned = clean_assistant_message(raw)
        self.assertEqual(cleaned["tool_calls"][0]["function"]["arguments"], '{"path": "a"}')

    def test_non_string_arguments_are_an_llm_error(self):
        raw = {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": 42}}]}
        with self.assertRaises(LLMError):
            clean_assistant_message(raw)

    def test_bad_usage_values_become_zero(self):
        parsed = parse_reply(self.good(usage={"prompt_tokens": -4, "completion_tokens": "many"}))
        self.assertEqual(parsed.usage["prompt_tokens"], 0)
        self.assertEqual(parsed.usage["completion_tokens"], 0)
        parsed = parse_reply(self.good(usage="?"))
        self.assertEqual(parsed.usage["prompt_tokens"], 0)

    def test_refusal_becomes_visible_text(self):
        parsed = parse_reply(self.good(choices=[{"message": {"role": "assistant", "content": None, "refusal": "I cannot do that."}}]))
        self.assertIn("refused", parsed.message["content"])

    def test_clean_message_drops_provider_extras(self):
        raw = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}], "reasoning": "secret", "refusal": None}
        cleaned = clean_assistant_message(raw)
        self.assertEqual(set(cleaned.keys()), {"role", "tool_calls"})

    def test_clean_message_never_returns_null_content_without_tools(self):
        self.assertEqual(clean_assistant_message({"role": "assistant", "content": None})["content"], "")


# ================================================================================
class ConfigTests(unittest.TestCase):
    def test_dotenv_parsing_keeps_existing_environment(self):
        env_file = Path(temp_dir(self)) / ".env"
        env_file.write_text('# comment\nTEST_ONE="a b"\nTEST_TWO=c=d\n\n')
        os.environ["TEST_ONE"] = "already"
        self.addCleanup(os.environ.pop, "TEST_ONE", None)
        self.addCleanup(os.environ.pop, "TEST_TWO", None)
        load_dotenv(env_file)
        self.assertEqual(os.environ["TEST_ONE"], "already")
        self.assertEqual(os.environ["TEST_TWO"], "c=d")

    def test_settings_missing_reports_placeholder_key(self):
        s = Settings(api_key="paste-your-key", base_url="x", model="m", workspace="w", max_steps=1)
        self.assertEqual(s.missing(), ["LLM_API_KEY"])


# ================================================================================
class ContextTests(unittest.TestCase):
    def build_history(self, rounds: int) -> list[dict]:
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "start"}]
        for i in range(rounds):
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": f"c{i}", "type": "function",
                                             "function": {"name": "read_file", "arguments": "{}"}}]})
            messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 400})
        messages.append({"role": "assistant", "content": "done"})
        return messages

    def test_cut_never_lands_on_a_tool_message(self):
        messages = self.build_history(10)
        for keep in range(1, len(messages)):
            cut = find_cut_index(messages, keep)
            self.assertNotEqual(messages[cut]["role"] if cut < len(messages) else "user", "tool")

    def test_trim_blanks_only_old_tool_results(self):
        messages = self.build_history(10)
        cut = find_cut_index(messages, keep_recent=4)
        trimmed = trim_old_tool_results(messages, keep_recent=4)
        old_tools = [m for m in trimmed[:cut] if m["role"] == "tool"]
        recent_tools = [m for m in trimmed[cut:] if m["role"] == "tool"]
        self.assertTrue(all(m["content"].startswith("[old tool output") for m in old_tools))
        self.assertTrue(all(m["content"] == "x" * 400 for m in recent_tools))
        self.assertEqual(len(trimmed), len(messages))

    def test_summarise_replaces_old_part_with_one_message(self):
        messages = self.build_history(10)
        fake = FakeLLM(["Goal: test. Progress: read ten files."])
        shrunk = summarise_older_messages(messages, fake, keep_recent=5)
        self.assertEqual(shrunk[0]["role"], "system")
        self.assertIn("Summary of the conversation", shrunk[1]["content"])
        self.assertLess(len(shrunk), len(messages))
        self.assertIn(shrunk[2]["role"], ("user", "assistant"))

    def test_agent_compacts_when_over_window(self):
        ws = make_workspace(self)
        registry = ToolRegistry(build_tools(ws))
        script = [[("list_files", {})]] * 6 + ["finished"]
        agent = Agent(llm=FakeLLM(script), tools=registry, system_prompt="sys", max_steps=20, context_window=300)
        events = []
        agent.run("go", on_event=events.append)
        self.assertIn("compaction", [e["type"] for e in events])
        self.assertEqual(check_message_order(agent.messages), [])


# ================================================================================
class FakeWfile:
    """Pretends to be the browser connection; breaks after `good_writes` writes."""

    def __init__(self, good_writes: int):
        self.good_writes = good_writes
        self.writes = 0
        self.events = []

    def write(self, data: bytes) -> None:
        if self.writes >= self.good_writes:
            raise BrokenPipeError("the tab was closed")
        self.writes += 1
        self.events.append(json.loads(data.decode("utf-8")[6:].strip()))

    def flush(self) -> None:
        pass


class EventStreamTests(unittest.TestCase):
    """A browser that disconnects must never leave the message list broken."""

    def run_with_disconnect_after(self, good_writes: int):
        ws = make_workspace(self)
        script = [[("list_files", {}), ("read_file", {"path": "missing.txt"})], "all done"]
        agent = Agent(llm=FakeLLM(script), tools=ToolRegistry(build_tools(ws)), system_prompt="sys")
        wfile = FakeWfile(good_writes)
        stream = server_module.EventStream(wfile)
        answer = agent.run("go", on_event=stream.send)
        self.assertEqual(answer, "all done")
        self.assertFalse(stream.alive)
        self.assertEqual(check_message_order(agent.messages), [])
        self.assertEqual([m["role"] for m in agent.messages],
                         ["system", "user", "assistant", "tool", "tool", "assistant"])
        return wfile.events

    def test_disconnect_after_model_reply(self):
        events = self.run_with_disconnect_after(3)   # user, model_call, model_reply, then it breaks
        self.assertEqual(events[-1]["type"], "model_reply")

    def test_disconnect_after_tool_call(self):
        events = self.run_with_disconnect_after(4)
        self.assertEqual(events[-1]["type"], "tool_call")

    def test_disconnect_after_first_result_of_two(self):
        events = self.run_with_disconnect_after(5)
        self.assertEqual(events[-1]["type"], "tool_result")


# ================================================================================
class SettingsForRunTests(unittest.TestCase):
    def test_model_only_override(self):
        run, problem = server_module.settings_for_run(make_settings(), {"model": "openai/gpt-5-nano"})
        self.assertIsNone(problem)
        self.assertEqual(run.model, "openai/gpt-5-nano")
        self.assertEqual(run.api_key, "test-key-not-real")

    def test_active_settings_are_never_mutated(self):
        active = make_settings()
        server_module.settings_for_run(active, {"model": "x", "api_key": "y"})
        self.assertEqual(active.model, "openai/gpt-5-mini")
        self.assertEqual(active.api_key, "test-key-not-real")

    def test_new_provider_url_requires_its_own_key(self):
        active = make_settings(base_url="https://api.openai.com/v1")
        run, problem = server_module.settings_for_run(active, {"base_url": "https://openrouter.ai/api/v1"})
        self.assertIsNone(run)
        self.assertIn("own API key", problem)

    def test_remote_http_is_refused_but_localhost_is_fine(self):
        active = make_settings()
        _, problem = server_module.settings_for_run(active, {"base_url": "http://example.com/v1", "api_key": "k"})
        self.assertIn("https", problem)
        old = server_module.ALLOW_ANY_PROVIDER
        server_module.ALLOW_ANY_PROVIDER = True
        self.addCleanup(setattr, server_module, "ALLOW_ANY_PROVIDER", old)
        run, problem = server_module.settings_for_run(active, {"base_url": "http://localhost:11434/v1", "api_key": "k"})
        self.assertIsNone(problem)
        self.assertEqual(run.base_url, "http://localhost:11434/v1")

    def test_unknown_provider_needs_the_explicit_option(self):
        active = make_settings()
        _, problem = server_module.settings_for_run(active, {"base_url": "https://api.example.com/v1", "api_key": "k"})
        self.assertIn("AGENT_ALLOW_ANY_PROVIDER", problem)


# ================================================================================
class ServerTests(unittest.TestCase):
    """The HTTP layer, driven by a real local socket and a fake model."""

    def start(self, llm, script_workspace=None):
        ws = script_workspace or make_workspace(self)
        agent = Agent(llm=llm, tools=ToolRegistry(build_tools(ws)), system_prompt="sys", max_steps=6)
        server_module.agent = agent
        server_module.settings = make_settings(workspace=ws.root)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_module.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.hostport = f"127.0.0.1:{httpd.server_address[1]}"
        return agent

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(self.hostport, timeout=15)
        data = None
        all_headers = dict(headers or {})
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
            all_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=data, headers=all_headers)
        response = connection.getresponse()
        payload = response.read().decode("utf-8")
        connection.close()
        return response.status, payload

    def chat_events(self, payload: str) -> list[dict]:
        return [json.loads(line[6:]) for line in payload.split("\n") if line.startswith("data: ")]

    def test_state_never_contains_the_api_key(self):
        self.start(FakeLLM(["hi"]))
        status, payload = self.request("GET", "/state")
        self.assertEqual(status, 200)
        self.assertNotIn("test-key-not-real", payload)
        self.assertNotIn("api_key", payload)

    def test_chat_streams_events_and_ends_with_done(self):
        agent = self.start(FakeLLM(["hello there"]))
        status, payload = self.request("POST", "/chat", {"text": "hi"})
        self.assertEqual(status, 200)
        kinds = [e["type"] for e in self.chat_events(payload)]
        self.assertEqual(kinds[0], "user")
        self.assertEqual(kinds[-2:], ["answer", "done"])
        self.assertEqual(len(agent.messages), 3)

    def test_unapproved_origin_gets_403_and_changes_nothing(self):
        agent = self.start(FakeLLM(["hi"]))
        status, payload = self.request("POST", "/chat", {"text": "hi"}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(len(agent.messages), 1)
        status, _ = self.request("POST", "/reset", {}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)

    def test_same_origin_and_listed_origin_are_allowed(self):
        self.start(FakeLLM(["a", "b"]))
        status, _ = self.request("POST", "/chat", {"text": "hi"}, {"Origin": "http://" + self.hostport})
        self.assertEqual(status, 200)
        status, _ = self.request("POST", "/chat", {"text": "hi"}, {"Origin": server_module.ALLOWED_ORIGINS[0]})
        self.assertEqual(status, 200)

    def test_malformed_requests_are_refused_without_touching_the_agent(self):
        agent = self.start(FakeLLM(["hi"]))
        cases = [
            (b"{\"text\": \"hi\"}", {"Content-Type": "text/plain"}),
            (b"[1, 2]", {}),
            (b"not json", {}),
            ({"text": ""}, {}),
            ({"text": 42}, {}),
            ({"text": "x" * (server_module.MAX_TEXT_CHARS + 1)}, {}),
        ]
        for body, headers in cases:
            status, payload = self.request("POST", "/chat", body, headers)
            self.assertEqual(status, 400, payload)
        self.assertEqual(len(agent.messages), 1)

    def test_reset_and_second_chat_during_a_run_are_refused(self):
        blocking = BlockingLLM(FakeLLM(["first answer"]))
        agent = self.start(blocking)
        results = {}

        def first_chat():
            results["first"] = self.request("POST", "/chat", {"text": "slow one"})

        thread = threading.Thread(target=first_chat)
        thread.start()
        self.assertTrue(blocking.started.wait(timeout=10))

        status, _ = self.request("POST", "/reset", {})
        self.assertEqual(status, 409)
        status, _ = self.request("POST", "/chat", {"text": "sneak in", "settings": {"model": "hijacked-model"}})
        self.assertEqual(status, 409)
        self.assertEqual(server_module.settings.model, "openai/gpt-5-mini")

        blocking.release.set()
        thread.join(timeout=10)
        self.assertEqual(results["first"][0], 200)
        self.assertEqual([m["role"] for m in agent.messages], ["system", "user", "assistant"])
        self.assertEqual(check_message_order(agent.messages), [])

    def test_model_override_becomes_active_only_after_a_successful_run(self):
        self.start(FakeLLM(["ok"]))
        status, payload = self.request("POST", "/chat", {"text": "hi", "settings": {"model": "openai/gpt-5-nano"}})
        self.assertEqual(status, 200)
        self.assertEqual(self.chat_events(payload)[-1]["model"], "openai/gpt-5-nano")
        status, payload = self.request("GET", "/state")
        self.assertEqual(json.loads(payload)["model"], "openai/gpt-5-nano")

    def test_invalid_model_is_a_clear_error_and_does_not_become_active(self):
        agent = self.start(FailingLLM("HTTP 400: The request was rejected. Usually a bad model id."))
        status, payload = self.request("POST", "/chat", {"text": "hi", "settings": {"model": "openai/gpt-6-luna"}})
        self.assertEqual(status, 200)
        events = self.chat_events(payload)
        self.assertTrue(any(e["type"] == "error" and "bad model id" in e["text"] for e in events))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(json.loads(self.request("GET", "/state")[1])["model"], "openai/gpt-5-mini")
        self.assertEqual(len(agent.messages), 1)

    def test_bad_provider_override_is_400_before_any_run(self):
        agent = self.start(FakeLLM(["ok"]))
        status, payload = self.request("POST", "/chat", {"text": "hi", "settings": {"base_url": "https://api.example.com/v1"}})
        self.assertEqual(status, 400)
        self.assertIn("own API key", payload)
        self.assertEqual(len(agent.messages), 1)
        self.assertFalse(server_module.run_lock.locked())

    def test_unexpected_exception_becomes_an_error_event_and_frees_the_lock(self):
        def broken(messages, tools):
            raise RuntimeError("a bug in a tool wrapper")

        self.start(broken)
        status, payload = self.request("POST", "/chat", {"text": "hi"})
        self.assertEqual(status, 200)
        events = self.chat_events(payload)
        self.assertTrue(any(e["type"] == "error" and "Unexpected error" in e["text"] for e in events))
        self.assertFalse(server_module.run_lock.locked())
        self.assertEqual(self.request("POST", "/reset", {})[0], 200)


if __name__ == "__main__":
    unittest.main()
