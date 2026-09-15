"""
Tests for the blueprint. Standard library unittest, no API key, no internet.

    python3 -m unittest -v

Each test is written so that its name tells you what would break.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import Settings, load_dotenv  # noqa: E402
from agent.fake_llm import FakeLLM  # noqa: E402
from agent.llm import LLMError, clean_assistant_message  # noqa: E402
from agent.loop import Agent  # noqa: E402
from agent.tools import Tool, ToolRegistry  # noqa: E402
from agent.workspace import Workspace, build_tools, truncate_output  # noqa: E402


def make_workspace() -> Workspace:
    return Workspace(tempfile.mkdtemp(prefix="agent-test-"))


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


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.ws = make_workspace()

    def test_paths_outside_root_are_refused(self):
        with self.assertRaises(ValueError):
            self.ws.safe_path("../../etc/passwd")

    def test_symlink_pointing_outside_is_refused(self):
        outside = tempfile.NamedTemporaryFile(delete=False)
        outside.write(b"secret")
        outside.close()
        os.symlink(outside.name, os.path.join(self.ws.root, "sneaky.txt"))
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
        self.assertEqual(open(os.path.join(self.ws.root, "a.txt")).read(), "1 two two")

    def test_run_command_reports_exit_code_and_stderr(self):
        text = self.ws.run_command("echo out; echo err 1>&2; exit 3")
        self.assertIn("exit code 3", text)
        self.assertIn("out", text)
        self.assertIn("[stderr]", text)

    def test_run_command_timeout_is_reported(self):
        self.assertIn("did not finish", self.ws.run_command("sleep 5", timeout_seconds=1))

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


class LoopTests(unittest.TestCase):
    def make_agent(self, script, max_steps=5):
        ws = make_workspace()
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

    def test_step_budget_stops_a_runaway_loop(self):
        endless = [[("list_files", {})]] * 10
        agent, _ = self.make_agent(endless, max_steps=3)
        answer = agent.run("loop forever")
        self.assertIn("Stopped after 3 steps", answer)

    def test_llm_error_becomes_the_answer(self):
        def broken(messages, tools):
            raise LLMError("HTTP 401: The API key was refused.")
        agent = Agent(llm=broken, tools=ToolRegistry(), system_prompt="sys")
        self.assertIn("401", agent.run("hi"))

    def test_reset_keeps_only_system_prompt(self):
        agent, _ = self.make_agent(["a", "b"])
        agent.run("1")
        agent.reset()
        self.assertEqual(len(agent.messages), 1)


class LLMHelperTests(unittest.TestCase):
    def test_clean_message_drops_provider_extras(self):
        raw = {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}], "reasoning": "secret", "refusal": None}
        cleaned = clean_assistant_message(raw)
        self.assertEqual(set(cleaned.keys()), {"role", "tool_calls"})

    def test_clean_message_never_returns_null_content_without_tools(self):
        self.assertEqual(clean_assistant_message({"role": "assistant", "content": None})["content"], "")

    def test_dotenv_parsing_keeps_existing_environment(self):
        env_file = Path(tempfile.mkdtemp()) / ".env"
        env_file.write_text('# comment\nTEST_ONE="a b"\nTEST_TWO=c=d\n\n')
        os.environ["TEST_ONE"] = "already"
        load_dotenv(env_file)
        self.assertEqual(os.environ["TEST_ONE"], "already")
        self.assertEqual(os.environ["TEST_TWO"], "c=d")

    def test_settings_missing_reports_placeholder_key(self):
        s = Settings(api_key="paste-your-key", base_url="x", model="m", workspace="w", max_steps=1)
        self.assertEqual(s.missing(), ["LLM_API_KEY"])


if __name__ == "__main__":
    unittest.main()


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
        from agent.context import find_cut_index
        messages = self.build_history(10)
        for keep in range(1, len(messages)):
            cut = find_cut_index(messages, keep)
            self.assertNotEqual(messages[cut]["role"] if cut < len(messages) else "user", "tool")

    def test_trim_blanks_only_old_tool_results(self):
        from agent.context import find_cut_index, trim_old_tool_results
        messages = self.build_history(10)
        cut = find_cut_index(messages, keep_recent=4)
        trimmed = trim_old_tool_results(messages, keep_recent=4)
        old_tools = [m for m in trimmed[:cut] if m["role"] == "tool"]
        recent_tools = [m for m in trimmed[cut:] if m["role"] == "tool"]
        self.assertTrue(all(m["content"].startswith("[old tool output") for m in old_tools))
        self.assertTrue(all(m["content"] == "x" * 400 for m in recent_tools))
        self.assertEqual(len(trimmed), len(messages))

    def test_summarise_replaces_old_part_with_one_message(self):
        from agent.context import summarise_older_messages
        messages = self.build_history(10)
        fake = FakeLLM(["Goal: test. Progress: read ten files."])
        shrunk = summarise_older_messages(messages, fake, keep_recent=5)
        self.assertEqual(shrunk[0]["role"], "system")
        self.assertIn("Summary of the conversation", shrunk[1]["content"])
        self.assertLess(len(shrunk), len(messages))
        self.assertEqual(shrunk[2]["role"] in ("user", "assistant"), True)

    def test_agent_compacts_when_over_window(self):
        ws = make_workspace()
        registry = ToolRegistry(build_tools(ws))
        script = [[("list_files", {})]] * 6 + ["finished"]
        fake = FakeLLM(script)
        agent = Agent(llm=fake, tools=registry, system_prompt="sys", max_steps=20, context_window=300)
        events = []
        agent.run("go", on_event=events.append)
        kinds = [e["type"] for e in events]
        self.assertIn("compaction", kinds)
