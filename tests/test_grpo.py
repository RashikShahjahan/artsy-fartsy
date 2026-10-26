"""Cookbook RL integration tests (no Docker/Tinker service required)."""

import asyncio
import base64
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from PIL import Image

from training.cookbook_env import (ArtsyGroupBuilder, BashTool, DockerSession,
                                   _preview_observation, configure_concurrency)
from training.grpo import make_cookbook_config, parser


def response(code=0, output=b"", stderr=b""):
    return SimpleNamespace(returncode=code, stdout=output, stderr=stderr)


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_is_the_only_terminating_bash_command(self):
        configure_concurrency(2)
        spec = {"id": "draw", "prompt": "Draw a circle", "seed": 42}
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession(spec, Path(root), MagicMock())
            tool = BashTool(session)

            async def docker(argv, **kwargs):
                if argv[0] == "run":
                    return response(output=b"id")
                if argv[0] == "exec":
                    return response(output=b"ok")
                return response()

            tools_module = ModuleType("tinker_cookbook.tool_use.tools")
            tools_module.simple_tool_result = lambda content, **kw: SimpleNamespace(
                should_stop=kw.get("should_stop", False),
                messages=[{"content": content, "tool_call_id": kw["call_id"]}])
            tools_module.error_tool_result = tools_module.simple_tool_result
            with patch("training.cookbook_env._docker", new=AsyncMock(side_effect=docker)), \
                 patch.dict(sys.modules, {"tinker_cookbook": ModuleType("tinker_cookbook"),
                     "tinker_cookbook.tool_use": ModuleType("tinker_cookbook.tool_use"),
                     "tinker_cookbook.tool_use.tools": tools_module}):
                result = await tool.run(SimpleNamespace(
                    arguments={"command": "echo hello"}, call_id="first"))
                self.assertFalse(result.should_stop)
                self.assertEqual(result.messages[0]["tool_call_id"], "first")
                submitted = await tool.run(SimpleNamespace(
                    arguments={"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
                    call_id="second"))
                self.assertTrue(submitted.should_stop)
                self.assertEqual(session.commands, [
                    "echo hello", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"])
                await session.close()
            self.assertIsNone(session.container)

    async def test_group_rewards_are_centered_only_within_the_prompt(self):
        spec = {"id": "draw", "prompt": "Draw a circle", "seed": 1}
        b = ArtsyGroupBuilder(spec, 2, 1, Path("/tmp"), MagicMock(), 128, 32768, True)
        trajectories = [SimpleNamespace(transitions=[SimpleNamespace(reward=r)])
                        for r in (0.0, 1.0)]
        rewards = await b.compute_group_rewards(trajectories, [])
        totals = [r + delta for r, (delta, _) in zip((0.0, 1.0), rewards)]
        self.assertAlmostEqual(totals[0], -1.0)
        self.assertAlmostEqual(totals[1], 1.0)

    async def test_fresh_trusted_render_and_correctness_grading(self):
        spec = {"id": "draw", "prompt": "Draw a circle", "seed": 1}
        png_file = BytesIO()
        Image.new("RGB", (1920, 1200)).save(png_file, "PNG")
        png = png_file.getvalue()
        judge = SimpleNamespace(complete=AsyncMock(return_value={
            "text": json.dumps({"score": 0.75, "reason": "Looks right"}),
            "finishReason": "stop"}))
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession(spec, Path(root), judge)
            session.container = "fake-container"

            async def docker(argv, **kwargs):
                if argv[0] == "exec":
                    return response(output=b"print('draw')\n")
                if argv[0] == "run":
                    self.assertIn("/opt/artsy/score_rl.py", argv)
                    self.assertEqual(kwargs["stdin"], b"print('draw')\n")
                    return response(output=json.dumps({
                        "feedback": {"ok": True}, "image": base64.b64encode(png).decode()
                    }).encode())
                return response()

            with patch("training.cookbook_env._docker", new=AsyncMock(side_effect=docker)):
                reward, metrics = await session.grade([])
            self.assertEqual(reward, 0.75)
            self.assertEqual(metrics["render_valid"], 1.0)
            self.assertEqual((Path(root) / "image.png").read_bytes(), png)
            self.assertEqual(json.loads((Path(root) / "result.json").read_text())
                             ["scores"]["adherence"], 0.75)


class PreviewTests(unittest.TestCase):
    def test_png_preview_becomes_image_part_with_feedback(self):
        output = BytesIO()
        Image.new("RGB", (24, 16), "navy").save(output, "PNG")
        encoded = base64.b64encode(output.getvalue()).decode()
        marker = ("<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>"
                  f"data:image/png;base64,{encoded}</MSWEA_MULTIMODAL_CONTENT>")
        payload = (json.dumps({"ok": True, "sourceHash": "f" * 64}) + "\n" + marker).encode()
        resize = ModuleType("transformers.models.qwen2_vl.image_processing_qwen2_vl")
        resize.smart_resize = lambda height, width, **kwargs: (height, width)
        modules = {name: ModuleType(name) for name in (
            "transformers", "transformers.models", "transformers.models.qwen2_vl")}
        modules[resize.__name__] = resize
        with patch.dict(sys.modules, modules), patch("training.cookbook_env.encoder") as get_encoder:
            get_encoder.return_value.image_processor.patch_size = 14
            get_encoder.return_value.image_processor.merge_size = 2
            parts, feedback = _preview_observation(payload, b"", 0)
        self.assertEqual(feedback["sourceHash"], "f" * 64)
        self.assertEqual([part["type"] for part in parts], ["text", "image"])
        self.assertEqual(parts[1]["image"].size, (24, 16))
        self.assertNotIn("data:image/png", parts[0]["text"])

    def test_text_feedback_without_preview(self):
        observation, feedback = _preview_observation(b'{"ok":false}\n', b"SyntaxError", 1)
        self.assertIn("Return code: 1", observation)
        self.assertIn("SyntaxError", observation)
        self.assertIsNone(feedback)

    def test_invalid_image_payload_is_rejected(self):
        marker = ("<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>"
                  "data:image/png;base64,AAAA</MSWEA_MULTIMODAL_CONTENT>")
        # The renderer's marker is not arbitrary text or a valid PNG.
        with self.assertRaises(ValueError):
            _preview_observation(marker.encode(), b"", 0)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_match_existing_agent_protocol(self):
        args = parser().parse_args([])
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.max_samples, 4)
        self.assertEqual(args.max_tokens, 4096)
        self.assertEqual(args.max_length, 32768)

    @unittest.skipUnless(importlib.util.find_spec("tinker_cookbook"),
                         "Install the pinned training dependencies")
    def test_cookbook_uses_ppo_kl_and_sft_initialization(self):
        args = parser().parse_args(["--max-steps", "1", "--clip-epsilon", "0.1"])
        args.output_dir = Path("/tmp/artsy-rl-example")
        plan = {"initial_state_path": "tinker://initial",
                "reference_sampler_path": "tinker://reference",
                "lora_rank": 32, "correctness_judge_model": "judge",
                "correctness_judge_url": "https://judge.example/v1"}
        config = make_cookbook_config(
            args, [{"id": "a", "prompt": "Draw a circle", "seed": 1}], plan, 1)
        self.assertEqual(config.loss_fn, "ppo")
        self.assertEqual(config.loss_fn_config["clip_low_threshold"], 0.9)
        self.assertEqual(config.loss_fn_config["clip_high_threshold"], 1.1)
        self.assertEqual(config.load_checkpoint_path, "tinker://initial")
        self.assertEqual(config.kl_reference_config.load_checkpoint_path, "tinker://reference")
        self.assertEqual(config.max_tokens, 32768)
        self.assertTrue(config.termination.zero_reward_on_limit)


if __name__ == "__main__":
    unittest.main()
