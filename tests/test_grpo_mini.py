"""Regression coverage for GRPO's mini-SWE/legacy harness boundary."""

import asyncio
import base64
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image


HAS_MINI = importlib.util.find_spec("minisweagent") is not None


def response(code=0, output=b"", stderr=b""):
    return SimpleNamespace(returncode=code, stdout=output, stderr=stderr)


@unittest.skipUnless(HAS_MINI, "Install the pinned mini-swe-agent dependency")
class MiniHarnessTests(unittest.IsolatedAsyncioTestCase):
    def test_mini_tool_and_sft_framing_match_the_training_export(self):
        from drawing.mini_agent import TOOLS, system_prompt
        from training.cookbook_mini import mini_initial_messages, mini_tool_spec
        spec = {"id": "a", "seed": 7, "prompt": "Draw a spiral"}
        renderer = Mock()
        renderer.create_conversation_prefix_with_tools.return_value = [
            {"role": "system", "content": "the mini-SWE system prefix"}]
        with patch("training.cookbook_mini.encoder", return_value=SimpleNamespace(renderer=renderer)):
            messages = mini_initial_messages(spec, "Qwen/model", "renderer", 1000, None)
        self.assertEqual(mini_tool_spec(), TOOLS[0].get("function", TOOLS[0]))
        self.assertEqual(mini_tool_spec()["name"], "bash")
        self.assertEqual(mini_tool_spec()["parameters"]["required"], ["command"])
        self.assertEqual(renderer.create_conversation_prefix_with_tools.call_args.kwargs,
                         {"system_prompt": system_prompt(7)})
        self.assertEqual(renderer.create_conversation_prefix_with_tools.call_args.args[0],
                         [mini_tool_spec()])
        self.assertEqual(messages[-1], {"role": "user", "content": spec["prompt"]})

    async def test_shell_executes_in_isolated_workspace_with_seed(self):
        from training.cookbook_mini import MiniDockerSession
        session = MiniDockerSession({"seed": 42}, Path("/tmp/test-rl"), {})
        session.container = "episode"
        session.start = AsyncMock()
        with patch("training.cookbook_mini._docker", new=AsyncMock(return_value=response(
                output=b"hello", stderr=b"warning"))) as docker:
            result = await session._shell("printf hello")
        args = docker.await_args.args[0]
        self.assertEqual(args[:8], ["exec", "-i", "-w", "/workspace", "-e", "ARTSY_SEED=42",
                                    "episode", "timeout"])
        self.assertEqual(args[-3:], ["bash", "-lc", "printf hello"])
        self.assertEqual(result, {"returncode": 0, "output": "hello\nwarning"})

    async def test_submitted_multifile_bundle_is_grader_input(self):
        from drawing.mini_protocol import EVENT_PREFIX, EVENT_SUFFIX, bundle_source
        from drawing.mini_agent import CONTRACT_VERSION
        from training.cookbook_mini import MiniDockerSession, MiniBashTool
        bundle = {"entrypoint": "src/main.py", "root": "/workspace",
                  "path": "/workspace/src/main.py",
                  "files": {"src/main.py": "import helper\nprint('hello')\n",
                            "src/helper.py": "print('helper')\n"}}
        event = EVENT_PREFIX + json.dumps({"kind": "submit", "bundle": bundle}) + EVENT_SUFFIX
        png_io = BytesIO()
        Image.new("RGB", (1920, 1200)).save(png_io, "PNG")
        png = png_io.getvalue()
        judge = {"hpsv3": SimpleNamespace(score=AsyncMock(return_value={
            "hpsv3_score": 1.0, "preference": 0.6}))}
        with tempfile.TemporaryDirectory() as root:
            session = MiniDockerSession({"seed": 4, "prompt": "Draw a circle"}, Path(root), judge)
            session.container = "episode"
            session.start = AsyncMock()

            async def docker(argv, **kwargs):
                if argv[0] == "exec":
                    return response(output=event.encode())
                if argv[0] == "run":
                    self.assertEqual(kwargs["stdin"], bundle_source(bundle).encode())
                    return response(output=json.dumps({"feedback": {"ok": True},
                        "image": base64.b64encode(png).decode()}).encode())
                return response()

            with patch("training.cookbook_mini._docker", new=AsyncMock(side_effect=docker)), \
                 patch("training.cookbook_env._docker", new=AsyncMock(side_effect=docker)):
                observation, terminal = await session.execute_bash("artsy-submit src/main.py")
                self.assertTrue(terminal)
                self.assertIn("Return code: 0", observation)
                self.assertTrue(session.submitted)
                self.assertEqual(session.drawing.entrypoint, "src/main.py")
                reward, metrics = await session.grade([])
            self.assertEqual(reward, 0.0)
            self.assertEqual(metrics["render_valid"], 1.0)
            saved = json.loads((Path(root) / "result.json").read_text())
            self.assertEqual(saved["trajectoryContractVersion"], CONTRACT_VERSION)
            self.assertEqual((Path(root) / "program.py").read_text(), bundle_source(bundle))
            self.assertEqual(session.tool_calls[0]["name"], "bash")
            judge["hpsv3"].score.assert_awaited_once()

    async def test_unsubmitted_workspace_source_does_not_score(self):
        from training.cookbook_mini import MiniDockerSession
        with tempfile.TemporaryDirectory() as root:
            session = MiniDockerSession({"prompt": "Draw", "seed": 3}, Path(root), {})
            session.drawing.source = "print('unsubmitted')"
            with patch("training.cookbook_env._docker", new=AsyncMock()) as docker:
                await session.grade([])
            docker.assert_not_awaited()
            self.assertEqual(session.result["scores"]["render_valid"], 0.0)

    def test_preview_image_is_converted_to_native_model_part(self):
        from training.cookbook_mini import _observation
        png_io = BytesIO()
        Image.new("RGB", (20, 15)).save(png_io, "PNG")
        url = "data:image/png;base64," + base64.b64encode(png_io.getvalue()).decode()
        encoded = SimpleNamespace(image_processor=object())
        with patch("training.cookbook_mini.encoder", return_value=encoded), \
             patch("training.cookbook_mini.resolve_model", return_value={"vision": "qwen"}), \
             patch("training.cookbook_mini.prepare_image", return_value="vision-tokens") as prepare:
            content = _observation([{"type": "text", "text": "Return code: 0"},
                                    {"type": "image_url", "image_url": {"url": url}}],
                                   "Qwen/model", "renderer", 5000, None)
        self.assertEqual(content[1], {"type": "image", "image": "vision-tokens"})
        self.assertEqual(prepare.call_args.kwargs, {"vision": "qwen"})
        self.assertEqual(prepare.call_args.args[0].size, (20, 15))

    def test_group_builder_constructs_mini_shell_tools(self):
        from training.cookbook_env import ArtsyGroupBuilder
        from training.cookbook_mini import MiniDockerSession, MiniBashTool
        group = ArtsyGroupBuilder({"id": "a", "prompt": "Draw", "seed": 1},
            1, 1, Path("/tmp/rl"), {}, 512, 32768, False, harness="mini-swe")
        group._resources = SimpleNamespace()
        agents = []
        class Agent:
            def __init__(self, **kwargs):
                agents.append(kwargs)
        class Env:
            def __init__(self, renderer, agent, **kwargs):
                self.renderer, self.agent, self.options = renderer, agent, kwargs
        class Limits:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        with patch("training.cookbook_mini.mini_initial_messages", return_value=[
             {"role": "user", "content": "Draw"}]), \
             patch("training.cookbook_env.encoder", return_value=SimpleNamespace(renderer="r")):
            environments = group._make_environments(Agent, Env, Limits)
        self.assertEqual(len(environments), 1)
        self.assertIsInstance(group.sessions[0], MiniDockerSession)
        self.assertEqual(len(agents[0]["tools"]), 1)
        self.assertIsInstance(agents[0]["tools"][0], MiniBashTool)
        self.assertEqual(agents[0]["max_turns"], 12)
        self.assertEqual(environments[0].options["rollout_limits"].max_turns, 12)

    def test_wrong_sft_harness_is_rejected(self):
        from drawing.documentation import api_documentation
        from training.data import digest
        from training.grpo import load_run, parser
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dataset = root / "prompts.jsonl"
            dataset.write_text(json.dumps({"id": "a", "prompt": "Draw", "seed": 1}) + "\n")
            checkpoint = root / "final.json"
            source = {"model": "Qwen/Qwen3.5-4B", "renderer": "qwen3_5_disable_thinking",
                      "lora_rank": 32, "tool_contract": "drawing-tools",
                      "api_doc_sha256": digest(api_documentation().encode())}
            checkpoint.write_text(json.dumps({"config": source,
                "state_path": "tinker://state", "sampler_path": "tinker://sampler"}))
            args = parser().parse_args(["--dataset", str(dataset), "--init-checkpoint", str(checkpoint)])
            with self.assertRaisesRegex(ValueError, "SFT checkpoint uses drawing-tools"):
                load_run(args)
            args.harness = "drawing-tools"
            _, config, _ = load_run(args)
            self.assertEqual(config["trajectory_contract_version"], 3)
            source["tool_contract"] = "mini-swe"
            checkpoint.write_text(json.dumps({"config": source,
                "state_path": "tinker://state", "sampler_path": "tinker://sampler"}))
            args.harness = "mini-swe"
            _, config, _ = load_run(args)
            self.assertEqual(config["trajectory_contract_version"], 4)


if __name__ == "__main__":
    unittest.main()
