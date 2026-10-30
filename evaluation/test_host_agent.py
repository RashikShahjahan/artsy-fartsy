"""Exercise the real host loop and Inspect recording without paid inference."""

import base64
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import (
    ChatCompletionChoice, ChatMessageAssistant, ContentImage, GenerateConfig,
    GenerateInput, ModelOutput, ModelUsage, get_model,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import TokenLimit
from PIL import Image

from evaluation.host_agent import SUBMIT, drawing_agent


def reply(command=None, *, calls=None, usage=20):
    calls = calls if calls is not None else ([ToolCall(
        id=f"call-{command}", function="bash", arguments={"command": command},
    )] if command is not None else None)
    return ModelOutput(model="mockllm/host-test", choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content="" if calls else "Done", tool_calls=calls),
        stop_reason="tool_calls" if calls else "stop",
    )], usage=ModelUsage(input_tokens=usage // 2, output_tokens=usage // 2, total_tokens=usage))


class HostAgentTests(unittest.TestCase):
    def run_agent(self, outputs, execute, *, callback=None, token_limit=32768, epochs=1,
                  max_calls=20, turn_limit=16, max_turns=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = get_model("mockllm/host-test", custom_outputs=outputs, memoize=False)
            sbox = SimpleNamespace(exec=AsyncMock(side_effect=execute))
            with patch("evaluation.host_agent.sandbox", return_value=sbox), \
                 patch("inspect_ai._util.appdirs.user_data_path", side_effect=lambda name: root / "data" / name), \
                 patch("inspect_ai._util.appdirs.user_cache_path", side_effect=lambda name: root / "cache" / name):
                logs = eval(Task(dataset=[Sample(id="drawing", input="Draw a circle")],
                    solver=drawing_agent("Use ArtCanvas.", GenerateConfig(max_tokens=4096,
                                         temperature=0.8, max_retries=0), callback,
                                         max_calls=max_calls, max_turns=max_turns),
                    token_limit=token_limit, turn_limit=turn_limit),
                    model=model, epochs=epochs, max_samples=2, log_dir=str(root / "logs"), display="none")
            self.assertEqual(logs[0].status, "success", logs[0].error)
            return list(read_eval_log(logs[0].location, resolve_attachments=True).samples), sbox

    def test_native_filter_repairs_preview_and_records_model_usage_images_and_commands(self):
        image = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(image, "PNG")
        url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
        outputs = iter([reply("write bad"), reply("preview bad"), reply("write fixed"),
                        reply("preview fixed"), reply(SUBMIT)])
        inputs = []

        async def native(model, messages, tools, tool_choice, config):
            inputs.append(list(messages))
            self.assertEqual(config.temperature, 0.8)
            self.assertEqual(config.max_tokens, 4096)
            self.assertEqual(tools[0].name, "bash")
            return next(outputs)

        async def execute(cmd, **kwargs):
            self.assertEqual(cmd[:5], ["timeout", "--signal=KILL", "65", "bash", "-c"])
            self.assertEqual(kwargs, {"cwd": "/workspace", "timeout": 67, "timeout_retry": False})
            stdout = ('{"ok":false,"exitCode":1}\n' if cmd[-1] == "preview bad" else "")
            if cmd[-1] == "preview fixed":
                stdout = ('{"ok":true,"exitCode":0}\n'
                    '<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>'
                    + url + '</MSWEA_MULTIMODAL_CONTENT>\n')
            return SimpleNamespace(stdout=stdout, stderr="", returncode=0, success=True)

        samples, sbox = self.run_agent([], execute, callback=native)
        sample = samples[0]
        self.assertIsNone(sample.error)
        self.assertEqual(sbox.exec.await_count, 5)
        self.assertEqual(len([e for e in sample.events if e.event == "model" and e.completed]), 5)
        self.assertEqual(len([e for e in sample.events if e.event == "tool"]), 5)
        self.assertEqual(sum(u.output_tokens for u in sample.model_usage.values()), 50)
        self.assertTrue(any(isinstance(p, ContentImage) and p.image == url
            for m in inputs[-1] for p in (m.content if isinstance(m.content, list) else [])))
        self.assertEqual(sample.metadata["harnessTiming"]["model_calls"], 5)
        self.assertTrue(sample.metadata["submitted"])
        self.assertGreaterEqual(sample.metadata["harnessTiming"]["agent_seconds"], 0)

    def test_plain_text_and_bad_arguments_are_repaired_without_host_execution(self):
        outputs = [reply(), reply(calls=[ToolCall(id="bad", function="bash", arguments={})]), reply(SUBMIT)]
        samples, sbox = self.run_agent(outputs, lambda *a, **k: SimpleNamespace(
            stdout="", stderr="", returncode=0, success=True))
        self.assertIsNone(samples[0].error)
        self.assertEqual(sbox.exec.await_count, 1)
        self.assertTrue(any(getattr(m, "error", None) for m in samples[0].messages))

    def test_marker_in_stdout_does_not_submit_and_later_commands_cannot_run_after_submission(self):
        outputs = [reply("printf marker"), reply(calls=[
            ToolCall(id="submit", function="bash", arguments={"command": SUBMIT}),
            ToolCall(id="late", function="bash", arguments={"command": "must not execute"}),
        ])]
        samples, sbox = self.run_agent(outputs, lambda *a, **k: SimpleNamespace(
            stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n", stderr="", returncode=0, success=True))
        self.assertIsNone(samples[0].error)
        self.assertEqual([c.args[0][-1] for c in sbox.exec.await_args_list], ["printf marker", SUBMIT])

    def test_native_usage_enforces_token_budget_before_command_execution(self):
        callback = AsyncMock(return_value=reply("must not execute", usage=100))
        async def native(*args):
            return await callback(*args)
        samples, sbox = self.run_agent([], None, callback=native, token_limit=50)
        self.assertEqual(samples[0].limit.type, "token")
        self.assertEqual(callback.await_count, 1)
        sbox.exec.assert_not_awaited()

    def test_output_budget_allows_repeated_input_history_and_reaches_submission(self):
        outputs = [reply("write program"), reply(SUBMIT)]
        for output in outputs:
            output.usage = ModelUsage(input_tokens=1000, output_tokens=10, total_tokens=1010)
        samples, sbox = self.run_agent(outputs, lambda *a, **k: SimpleNamespace(
            stdout="", stderr="", returncode=0, success=True),
            token_limit=TokenLimit(tokens=50, type="output"))
        self.assertIsNone(samples[0].error)
        self.assertIsNone(samples[0].limit)
        self.assertEqual(sbox.exec.await_count, 2)
        self.assertEqual(sum(u.output_tokens for u in samples[0].model_usage.values()), 20)

    def test_no_aggregate_budget_allows_output_beyond_the_previous_cap(self):
        outputs = [reply(f"write part {i}") for i in range(8)] + [reply(SUBMIT)]
        for output in outputs:
            output.usage = ModelUsage(input_tokens=10000, output_tokens=4000, total_tokens=14000)
        samples, sbox = self.run_agent(outputs, lambda *a, **k: SimpleNamespace(
            stdout="", stderr="", returncode=0, success=True), token_limit=None)
        self.assertIsNone(samples[0].error)
        self.assertIsNone(samples[0].limit)
        self.assertEqual(sbox.exec.await_count, 9)
        self.assertEqual(sum(u.output_tokens for u in samples[0].model_usage.values()), 36000)

    def test_turn_budget_stops_repeated_text_answers(self):
        samples, sbox = self.run_agent([reply() for _ in range(17)], None)
        self.assertEqual(len([m for m in samples[0].messages if m.role == "assistant"]), 16)
        sbox.exec.assert_not_awaited()
        self.assertFalse(samples[0].metadata["submitted"])

    def test_announced_turn_limit_counts_text_answers_and_stops_at_25(self):
        inputs = []
        async def native(model, messages, tools, tool_choice, config):
            inputs.append(list(messages))
            return reply()
        samples, _ = self.run_agent([], None, callback=native, max_calls=None,
                                    turn_limit=None, max_turns=25)
        self.assertEqual(len(inputs), 25)
        self.assertIn('You have 25 model turns', inputs[0][1].text)
        for turn, messages in enumerate(inputs, 1):
            self.assertTrue(messages[-1].text.startswith(f'Turn {turn} of 25.'))
        self.assertIn('final turn', inputs[-1][-1].text)
        self.assertTrue(samples[0].metadata['turnLimitReached'])

    def test_submission_on_turn_25_is_executed(self):
        outputs = [reply() for _ in range(24)] + [reply(SUBMIT)]
        samples, sbox = self.run_agent(outputs, lambda *a, **k: SimpleNamespace(
            stdout='COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n', stderr='', returncode=0,
            success=True), max_calls=None, turn_limit=None, max_turns=25)
        self.assertEqual(samples[0].metadata['harnessTiming']['model_calls'], 25)
        self.assertTrue(samples[0].metadata['submitted'])
        self.assertFalse(samples[0].metadata['turnLimitReached'])
        self.assertEqual(sbox.exec.await_count, 1)

    def test_operation_budget_cannot_be_bypassed_with_multiple_calls_per_turn(self):
        from drawing.config import LIMITS
        calls = [ToolCall(id=f"call-{i}", function="bash", arguments={"command": f"write {i}"})
                 for i in range(LIMITS["maxCalls"])]
        calls.append(ToolCall(id="submit", function="bash", arguments={"command": SUBMIT}))
        samples, sbox = self.run_agent([reply(calls=calls)], lambda *a, **k: SimpleNamespace(
            stdout="", stderr="", returncode=0, success=True))
        self.assertEqual(sbox.exec.await_count, LIMITS["maxCalls"])
        self.assertEqual(samples[0].metadata["environmentCalls"], LIMITS["maxCalls"])
        self.assertFalse(samples[0].metadata["submitted"])
        self.assertTrue(any("operation budget exhausted" in m.text
                            for m in samples[0].messages if m.role == "tool"))

    def test_unlimited_calls_can_submit_after_more_than_twenty_turns(self):
        outputs = [reply(f"write {i}") for i in range(25)] + [reply(SUBMIT)]
        samples, sbox = self.run_agent(outputs, lambda *a, **k: SimpleNamespace(
            stdout="", stderr="", returncode=0, success=True), max_calls=None, turn_limit=None)
        self.assertEqual(sbox.exec.await_count, 26)
        self.assertTrue(samples[0].metadata["submitted"])
        self.assertIsNone(samples[0].limit)

    def test_unlimited_calls_allows_more_than_twenty_operations_in_one_turn(self):
        calls = [ToolCall(id=f"call-{i}", function="bash", arguments={"command": f"write {i}"})
                 for i in range(25)]
        calls.append(ToolCall(id="submit", function="bash", arguments={"command": SUBMIT}))
        samples, sbox = self.run_agent([reply(calls=calls)], lambda *a, **k: SimpleNamespace(
            stdout="", stderr="", returncode=0, success=True), max_calls=None, turn_limit=None)
        self.assertEqual(sbox.exec.await_count, 26)
        self.assertTrue(samples[0].metadata["submitted"])

    def test_filter_can_modify_request_and_fall_back_to_the_configured_provider(self):
        seen = []

        def provider(messages, tools, choice, config):
            seen.append(config.temperature)
            return reply(SUBMIT)

        async def modify(model, messages, tools, choice, config):
            return GenerateInput(messages, tools, choice, config.model_copy(update={"temperature": 0.3}))

        samples, sbox = self.run_agent(provider, lambda *a, **k: SimpleNamespace(
            stdout="", stderr="", returncode=0, success=True), callback=modify)
        self.assertIsNone(samples[0].error)
        self.assertEqual(seen, [0.3])
        self.assertEqual(sbox.exec.await_count, 1)

    def test_command_timeout_is_feedback_and_allows_next_turn(self):
        async def execute(cmd, **kwargs):
            if cmd[-1] == "hang":
                raise TimeoutError()
            return SimpleNamespace(stdout="", stderr="", returncode=0, success=True)

        samples, sbox = self.run_agent([reply("hang"), reply(SUBMIT)], execute)
        self.assertIsNone(samples[0].error)
        self.assertEqual(sbox.exec.await_count, 2)
        self.assertTrue(any("timed out" in m.text for m in samples[0].messages if m.role == "tool"))

    @unittest.skipUnless(os.getenv("ARTSY_DOCKER_SMOKE") == "1", "set ARTSY_DOCKER_SMOKE=1 for Docker")
    def test_docker_tools_are_isolated_and_fresh_scoring_preserves_multimodal_traces(self):
        from evaluation.harness import drawing_task
        from training.data import write_command

        def policy(messages, tools, choice, config):
            turn = sum(m.role == "assistant" for m in messages)
            if turn == 0:
                source = ('from artcanvas import ArtCanvas\n'
                          'with ArtCanvas(1920,1200) as c:\n'
                          ' c.set_color(COLOR)\n c.draw_circle(960,600,200,True)\n')
                source = source.replace("COLOR", "1,0,0" if "red circle" in messages[1].text else "0,0,1")
                # The execution image has no agent or model SDK; only drawing tools.
                command = ('python -c "import importlib.util as u; '
                    "assert all(u.find_spec(p) is None for p in ['minisweagent','litellm','openai'])\"\n"
                    + write_command(source))
            elif turn == 1:
                command = "python /opt/artsy/render.py --seed 42 --output /workspace/preview --preview < /workspace/program.py"
            else:
                self.assertTrue(any(isinstance(p, ContentImage) for m in messages
                    for p in (m.content if isinstance(m.content, list) else [])))
                command = SUBMIT
            return reply(command)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specs = [{"id": color, "prompt": f"Draw a {color} circle", "seed": 42}
                     for color in ("red", "blue")]
            model = get_model("mockllm/docker-host-test", custom_outputs=policy, memoize=False)
            logs = eval(drawing_task(specs, str(root / "artifacts")), model=model,
                        max_samples=2, log_dir=str(root / "logs"), display="none")
            self.assertEqual(logs[0].status, "success", logs[0].error)
            samples = list(read_eval_log(logs[0].location, resolve_attachments=True).samples)
            self.assertEqual(len(samples), 2)
            for sample in samples:
                self.assertIsNone(sample.error)
                self.assertEqual(sample.scores["drawing_score"].value["render_valid"], 1)
                self.assertEqual(sample.metadata["harnessTiming"]["model_calls"], 3)
                self.assertEqual(len([e for e in sample.events if e.event == "tool"]), 3)
            results = [json.loads((root / "artifacts" / f"{color}-1" / "result.json").read_text())
                       for color in ("red", "blue")]
            self.assertNotEqual(results[0]["imageHash"], results[1]["imageHash"])
            self.assertTrue(all(r["agent"] == "artsy-host" for r in results))
            print("Docker host-harness smoke timings:", json.dumps([r["harnessTiming"] for r in results]))


if __name__ == "__main__":
    unittest.main()
