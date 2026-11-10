"""RLVR verifier, scoring isolation, and SFT -> RLVR -> RLHF handoff tests."""

import base64
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

from drawing.config import MODEL_ID
from drawing.documentation import api_documentation
from training.cookbook_env import ArtsyGroupBuilder, DockerSession
from training.data import digest
from training.grpo import load_run, parser
from training.rlvr_reward import RLVR_ALGORITHM, verified_reward
from training.sft import RENDERER


class VerifierTests(unittest.TestCase):
    def test_clean_submission_receives_zero_not_a_positive_bonus(self):
        value = verified_reward(submitted=True, render_valid=True,
                                valid_tool_calls=2, invalid_tool_calls=0,
                                execution_attempts=2, execution_errors=0)
        self.assertEqual(value, {
            "image_verified": 1.0, "tool_protocol": 1.0,
            "execution_success": 1.0, "image_penalty": 0.0,
            "tool_penalty": 0.0, "execution_penalty": 0.0,
            "reward": 0.0})

    def test_missing_image_and_missing_tool_calls_are_penalized(self):
        no_work = verified_reward(
            submitted=False, render_valid=False, valid_tool_calls=0,
            invalid_tool_calls=0, execution_attempts=0, execution_errors=0)
        self.assertEqual(no_work["reward"], -1.025)
        self.assertEqual(no_work["image_penalty"], -1.0)
        self.assertEqual(no_work["tool_penalty"], -0.025)
        self.assertEqual(no_work["execution_penalty"], 0.0)

        unsubmitted = verified_reward(
            submitted=False, render_valid=True, valid_tool_calls=5,
            invalid_tool_calls=0, execution_attempts=5, execution_errors=0)
        self.assertEqual(unsubmitted["image_verified"], 0.0)
        self.assertEqual(unsubmitted["reward"], -1.0)

    def test_each_tool_and_execution_failure_adds_a_negative_penalty(self):
        value = verified_reward(submitted=True, render_valid=True,
                                valid_tool_calls=3, invalid_tool_calls=2,
                                execution_attempts=3, execution_errors=1)
        self.assertEqual(value["image_penalty"], 0.0)
        self.assertEqual(value["tool_penalty"], -0.05)
        self.assertEqual(value["execution_penalty"], -0.025)
        self.assertAlmostEqual(value["reward"], -0.075)
        self.assertLessEqual(value["reward"], 0.0)

        invalid = verified_reward(submitted=True, render_valid=False,
                                  valid_tool_calls=1, invalid_tool_calls=1,
                                  execution_attempts=2, execution_errors=1)
        self.assertAlmostEqual(invalid["reward"], -1.05)

    def test_successful_commands_cannot_dilute_prior_errors(self):
        def score(successes):
            return verified_reward(
                submitted=True, render_valid=True,
                valid_tool_calls=successes + 1, invalid_tool_calls=1,
                execution_attempts=successes + 1, execution_errors=1)["reward"]
        self.assertEqual(score(0), -0.05)
        self.assertEqual(score(99), -0.05)

    def test_error_categories_are_capped_independently(self):
        for protocol_errors, execution_errors, tool_penalty, execution_penalty in (
            (7, 1, -0.10, -0.025),
            (1, 7, -0.025, -0.10),
            (7, 7, -0.10, -0.10),
        ):
            with self.subTest(protocol_errors=protocol_errors, execution_errors=execution_errors):
                value = verified_reward(
                    submitted=False, render_valid=False, valid_tool_calls=1,
                    invalid_tool_calls=protocol_errors, execution_attempts=7,
                    execution_errors=execution_errors)
                self.assertEqual(value["tool_penalty"], tool_penalty)
                self.assertEqual(value["execution_penalty"], execution_penalty)
                self.assertAlmostEqual(value["reward"], -1.0 + tool_penalty + execution_penalty)

    def test_invalid_evidence_is_rejected(self):
        for counts in (
            dict(valid_tool_calls=1, invalid_tool_calls=0,
                 execution_attempts=1, execution_errors=2),
            dict(valid_tool_calls=-1, invalid_tool_calls=0,
                 execution_attempts=0, execution_errors=0),
            dict(valid_tool_calls=True, invalid_tool_calls=0,
                 execution_attempts=0, execution_errors=0),
        ):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                verified_reward(submitted=True, render_valid=True, **counts)


class CheckpointHandoffTests(unittest.TestCase):
    def test_rlvr_requires_sft_and_rlhf_inherits_sft_reference(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dataset = root / "prompts.jsonl"
            dataset.write_text(json.dumps({
                "id": "one", "prompt": "Draw a circle", "seed": 1}) + "\n")
            checkpoint = root / "final.json"
            sft_config = {"model": MODEL_ID, "renderer": RENDERER, "lora_rank": 32,
                          "tool_contract": "mini-swe",
                          "api_doc_sha256": digest(api_documentation().encode())}
            checkpoint.write_text(json.dumps({
                "config": sft_config, "state_path": "tinker://sft/state",
                "sampler_path": "tinker://sft/sampler"}))
            args = parser().parse_args([
                "--reward-mode", "rlvr", "--dataset", str(dataset),
                "--init-checkpoint", str(checkpoint)])
            _, plan, _ = load_run(args)
            self.assertEqual(plan["algorithm"], RLVR_ALGORITHM)
            self.assertEqual(plan["reward_mode"], "rlvr")
            self.assertEqual(plan["group_size"], 2)
            self.assertEqual(plan["reference_sampler_path"], "tinker://sft/sampler")
            self.assertIsNone(plan["hpsv3_model"])
            self.assertEqual(plan["checkpoint_tool_contract"], "mini-swe")

            rlvr_source = {**plan, "algorithm": RLVR_ALGORITHM}
            checkpoint.write_text(json.dumps({
                "config": rlvr_source, "state_path": "tinker://rlvr/state",
                "sampler_path": "tinker://rlvr/sampler"}))
            with self.assertRaisesRegex(ValueError, "compatible SFT"):
                load_run(args)
            args.reward_mode = "rlhf"
            _, rlhf, _ = load_run(args)
            self.assertEqual(rlhf["initial_state_path"], "tinker://rlvr/state")
            self.assertEqual(rlhf["reference_sampler_path"], "tinker://sft/sampler")
            self.assertEqual(rlhf["reference_model"], plan["reference_model"])
            self.assertEqual(rlhf["reward_mode"], "rlhf")
            self.assertEqual(rlhf["group_size"], 4)
            self.assertEqual(rlhf["algorithm"], "tinker-cookbook-rlhf-v8")
            rlvr_source["harness"] = "drawing-tools"
            checkpoint.write_text(json.dumps({
                "config": rlvr_source, "state_path": "tinker://rlvr/state",
                "sampler_path": "tinker://rlvr/sampler"}))
            with self.assertRaises(ValueError):
                load_run(args)

    def test_canonical_pool_uses_disjoint_stage_splits_before_limit(self):
        from training.rl_split import SPLIT_VERSION
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            dataset = root / "rl" / "prompts.jsonl"
            dataset.parent.mkdir()
            records = [
                {"id": f"parti-{i:04d}", "prompt": f"Draw scene {i}", "seed": i,
                 "source": {"category": f"cat-{i % 4}", "challenge": f"level-{i % 3}"}}
                for i in range(327)]
            dataset.write_text("".join(json.dumps(record) + "\n" for record in records))
            checkpoint = root / "sft.json"
            checkpoint.write_text(json.dumps({
                "config": {"model": MODEL_ID, "renderer": RENDERER, "lora_rank": 32,
                           "tool_contract": "mini-swe",
                           "api_doc_sha256": digest(api_documentation().encode())},
                "state_path": "tinker://sft/state",
                "sampler_path": "tinker://sft/sampler"}))
            args = parser().parse_args([
                "--dataset", str(dataset), "--init-checkpoint", str(checkpoint)])
            with patch("training.rl_split.DEFAULT_RL_DATASET", dataset):
                args.reward_mode = "rlvr"
                verifier_specs, verifier_config, verifier_updates = load_run(args)
                args.reward_mode = "rlhf"
                preference_specs, preference_config, preference_updates = load_run(args)
                self.assertEqual((len(verifier_specs), len(preference_specs)), (227, 100))
                self.assertEqual((verifier_updates, preference_updates), (29, 13))
                self.assertEqual(verifier_config["group_size"], 2)
                self.assertEqual(preference_config["group_size"], 4)
                args.group_size = 6
                _, override, _ = load_run(args)
                self.assertEqual(override["group_size"], 6)
                args.group_size = None
                self.assertEqual(verifier_config["rl_stage_split_version"], SPLIT_VERSION)
                self.assertEqual(preference_config["rl_stage_split_version"], SPLIT_VERSION)
                self.assertFalse(
                    {item["id"] for item in verifier_specs}
                    & {item["id"] for item in preference_specs})
                args.limit = 5
                limited, config, updates = load_run(args)
                self.assertEqual(limited, preference_specs[:5])
                self.assertEqual(config["prompt_ids"], [row["id"] for row in limited])
                self.assertEqual(updates, 1)

    def test_rlvr_forbids_legacy_tools(self):
        with tempfile.TemporaryDirectory() as root:
            dataset = Path(root) / "prompts.jsonl"
            dataset.write_text(json.dumps({"id": "one", "prompt": "Draw", "seed": 1}) + "\n")
            args = parser().parse_args([
                "--reward-mode", "rlvr", "--harness", "drawing-tools",
                "--dataset", str(dataset)])
            with self.assertRaisesRegex(ValueError, "mini-swe"):
                load_run(args)


class RLVRDatasetTests(unittest.IsolatedAsyncioTestCase):
    async def test_rlvr_does_not_construct_hpsv3_client(self):
        from training.cookbook_dataset import ArtsyDatasetBuilder
        builder = ArtsyDatasetBuilder(
            specs=({"id": "one", "prompt": "Draw", "seed": 1},),
            output_dir="/tmp/artsy-rlvr-test", batch_size=1, group_size=2,
            epochs=1, seed=42, max_tokens=None, max_length=32768,
            model=MODEL_ID, renderer=RENDERER, revision=None,
            normalize_rewards=False, hpsv3_model=None,
            hpsv3_url="http://127.0.0.1:8090", reward_mode="rlvr",
            harness="mini-swe")
        with patch("training.cookbook_dataset.HPSv3Client") as client:
            dataset, evaluation = await builder()
        client.assert_not_called()
        self.assertEqual(dataset.judge, {})
        self.assertIsNone(evaluation)
        group, = dataset.get_batch(0)
        self.assertEqual(group.reward_mode, "rlvr")


class DockerVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_trusted_final_image_scores_without_visual_judge(self):
        png_io = BytesIO()
        Image.new("RGB", (1920, 1200)).save(png_io, "PNG")
        png = png_io.getvalue()
        spec = {"id": "one", "prompt": "Draw a circle", "seed": 1}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = DockerSession(spec, root, {}, reward_mode="rlvr")
            session.submitted = True
            session.tool_calls.append({"name": "bash", "arguments": {"command": "artsy-submit"}})
            session.invalid_tool_calls = 1
            session.execution_attempts = 3
            session.execution_errors = 1
            async def docker(argv, **kwargs):
                if argv[0] == "run":
                    return SimpleNamespace(returncode=0, stderr=b"",
                        stdout=json.dumps({"feedback": {"ok": True},
                          "image": base64.b64encode(png).decode()}).encode())
                return SimpleNamespace(returncode=0, stderr=b"", stdout=b"")
            with patch.object(session, "_read_submission",
                              new=AsyncMock(return_value=("print('draw')", None))), \
                 patch("training.cookbook_env._docker", new=AsyncMock(side_effect=docker)):
                await session.grade([])
            self.assertNotIn("judgment", session.result)
            self.assertEqual(session.result["scores"]["render_valid"], 1.0)
            group = ArtsyGroupBuilder(spec, 1, 1, root, {}, 128, 32768, False,
                                      harness="mini-swe", reward_mode="rlvr")
            group.sessions = [session]
            adjustment, = await group.compute_group_rewards(
                [SimpleNamespace(transitions=[SimpleNamespace(reward=0.1)])], [])
            expected = -0.05
            self.assertAlmostEqual(adjustment[0] + 0.1, expected)
            saved = json.loads((root / "result.json").read_text())
            self.assertEqual(saved["invalidToolCalls"], 1)
            self.assertEqual(saved["executionErrors"], 1)
            self.assertAlmostEqual(saved["scores"]["reward"], expected)

    async def test_unsubmitted_source_is_not_rendered_or_credited_as_image(self):
        spec = {"id": "one", "prompt": "Draw", "seed": 1}
        with tempfile.TemporaryDirectory() as directory:
            session = DockerSession(spec, Path(directory), {}, reward_mode="rlvr")
            session.tool_calls.append({"name": "bash", "arguments": {"command": "echo hello"}})
            session.execution_attempts = 1
            with patch.object(session, "_read_submission",
                              new=AsyncMock(return_value=("print('draw')", None))), \
                 patch("training.cookbook_env._docker", new=AsyncMock()) as docker:
                await session.grade([])
            docker.assert_not_awaited()
            group = ArtsyGroupBuilder(spec, 1, 1, Path(directory), {}, 128, 32768,
                                      False, harness="mini-swe", reward_mode="rlvr")
            group.sessions = [session]
            adjustment, = await group.compute_group_rewards(
                [SimpleNamespace(transitions=[])], [])
            self.assertEqual(adjustment[0], -1.0)
            self.assertEqual(session.result["scores"]["image_verified"], 0.0)


@unittest.skipUnless(importlib.util.find_spec("minisweagent"), "mini-SWE not installed")
class MiniExecutionEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_native_bash_call_is_dispatched(self):
        from tinker_cookbook.renderers import ToolCall
        from training.cookbook_env import configure_concurrency

        configure_concurrency(1, 1)
        with tempfile.TemporaryDirectory() as root:
            builder = ArtsyGroupBuilder({"id": "x", "seed": 1}, 1, 1, Path(root), {},
                                        None, 262144, False, harness="mini-swe",
                                        reward_mode="rlvr")
            self.addAsyncCleanup(builder.cleanup)
            renderer = SimpleNamespace(get_stop_sequences=lambda: [2])
            with patch("training.cookbook_env.encoder", return_value=SimpleNamespace(renderer=renderer)), \
                 patch("training.cookbook_mini.mini_initial_messages", return_value=[]):
                wrapped, = await builder.make_envs()
            session, = builder.sessions
            session.execute_bash = AsyncMock(return_value=("ok", False))
            call = ToolCall(id="bash-1", function=ToolCall.FunctionBody(
                name="bash", arguments=json.dumps({"command": "echo ok"})))
            result = await wrapped.message_env.step(
                {"role": "assistant", "content": "", "tool_calls": [call]})

            session.execute_bash.assert_awaited_once_with("echo ok")
            self.assertEqual(session.invalid_tool_calls, 0)
            self.assertFalse(result.episode_done)
            self.assertEqual(result.next_messages[-1]["role"], "tool")

    async def test_nonzero_exits_failed_previews_and_bad_arguments_are_recorded(self):
        from training.cookbook_mini import MiniDockerSession
        session = MiniDockerSession({"seed": 1}, Path("/tmp/rlvr-test"), {},
                                     reward_mode="rlvr")
        session.drawing.execute = AsyncMock(side_effect=[
            {"returncode": 0, "content": [{"type": "text", "text": "ok"}],
             "previewEvents": []},
            {"returncode": 0, "content": [{"type": "text", "text": "failed"}],
             "previewEvents": [{"ok": False}]},
            {"returncode": 1, "content": [{"type": "text", "text": "error"}],
             "previewEvents": []},
        ])
        with patch("training.cookbook_mini._observation", return_value="observed"):
            for command in ("echo ok", "artsy-preview", "python fail.py"):
                await session.execute_bash(command)
        self.assertEqual(session.execution_attempts, 3)
        self.assertEqual(session.execution_errors, 2)
        self.assertEqual(len(session.tool_calls), 3)
        await session.execute_bash(" ")
        self.assertEqual(session.invalid_tool_calls, 1)


if __name__ == "__main__":
    unittest.main()
